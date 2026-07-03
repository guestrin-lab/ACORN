/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#include <faiss/impl/ACORN.h>

#include <string>

#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/DistanceComputer.h>
#include <faiss/impl/IDSelector.h>

// added
#include <sys/time.h>
#include <stdio.h>
#include <iostream>
#include <math.h>  
#include <unordered_map>
#include <iostream>
#include <fstream>
/*******************************************************
 * Added for debugging
 *******************************************************/

const int debugFlag = 0;

// const int debugSearchFlag =  std::atoi(std::getenv(debugSearchFlag));
const char* debugSearchFlagEnv = std::getenv("debugSearchFlag");
int debugSearchFlag = debugSearchFlagEnv ? std::atoi(debugSearchFlagEnv) : 0;


void debugTime() {
	if (debugFlag || debugSearchFlag) {
        struct timeval tval;
        gettimeofday(&tval, NULL);
        struct tm *tm_info = localtime(&tval.tv_sec);
        char timeBuff[25] = "";
        strftime(timeBuff, 25, "%H:%M:%S", tm_info);
        char timeBuffWithMilli[50] = "";
        sprintf(timeBuffWithMilli, "%s.%06ld ", timeBuff, tval.tv_usec);
        std::string timestamp(timeBuffWithMilli);
		std::cout << timestamp << std::flush;
    }
}

#define debug(fmt, ...) \
    do { \
        if (debugFlag == 1) { \
            fprintf(stdout, "" fmt, __VA_ARGS__);\
        } \
        if (debugFlag == 2) { \
            debugTime(); \
            fprintf(stdout, "%s:%d:%s(): " fmt, __FILE__, __LINE__, __func__, __VA_ARGS__); \
        } \
    } while (0)

// same as debug but for search debugging only
#define debug_search(fmt, ...) \
    do { \
        if (debugSearchFlag == 1) { \
            fprintf(stdout, "" fmt, __VA_ARGS__);\
        } \
        if (debugSearchFlag == 2) { \
            fprintf(stdout, "%d:%s(): " fmt, __LINE__, __func__, __VA_ARGS__); \
        } \
        if (debugSearchFlag == 3) { \
            debugTime(); \
            fprintf(stdout, "%s:%d:%s(): " fmt, __FILE__, __LINE__, __func__, __VA_ARGS__); \
        } \
    } while (0)


double elapsed() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}



namespace faiss {




/**************************************************************
 * ACORN structure implementation
 **************************************************************/

int ACORN::nb_neighbors(int layer_no) const {
    return cum_nneighbor_per_level[layer_no + 1] -
            cum_nneighbor_per_level[layer_no];
}

void ACORN::set_nb_neighbors(int level_no, int n) {
    FAISS_THROW_IF_NOT(levels.size() == 0);
    int cur_n = nb_neighbors(level_no);
    for (int i = level_no + 1; i < cum_nneighbor_per_level.size(); i++) {
        cum_nneighbor_per_level[i] += n - cur_n;
    }
}

int ACORN::cum_nb_neighbors(int layer_no) const {
    return cum_nneighbor_per_level[layer_no];
}

void ACORN::neighbor_range(idx_t no, int layer_no, size_t* begin, size_t* end)
        const {
    size_t o = offsets[no];
    // debug("offset o: %ld\n", o);
    *begin = o + cum_nb_neighbors(layer_no);
    // debug("begin: %ln\n", begin);
    *end = o + cum_nb_neighbors(layer_no + 1);
    // debug("end: %ln\n", end);
}



ACORN::ACORN(int M, int gamma, std::vector<int>& metadata, int M_beta) : rng(12345) {
    set_default_probas(M, 1.0 / log(M), M_beta, gamma);
    max_level = -1;
    entry_point = -1;
    efSearch = 16;
    efConstruction = M * gamma; //added gamma
    upper_beam = 1;
    this->gamma = gamma;
    this->metadata = metadata.data();
    this->M = M;
    this->M_beta = M_beta;
    // gamma = gamma;
    offsets.push_back(0);
    for (int i = 0; i < assign_probas.size(); i++) nb_per_level.push_back(0);
}

int ACORN::random_level() {
    double f = rng.rand_float();
    // could be a bit faster with bissection
    for (int level = 0; level < assign_probas.size(); level++) {
        if (f < assign_probas[level]) {
            return level;
        }
        f -= assign_probas[level];
    }
    // happens with exponentially low probability
    return assign_probas.size() - 1;
}

void ACORN::set_default_probas(int M, float levelMult, int M_beta, int gamma) {
    int nn = 0;
    cum_nneighbor_per_level.push_back(0);
    // printf("---set_default_probas: gamma: %d\n", this->gamma);
    // printf("---set_default_probas: gamma: %d\n", gamma);
    if (M_beta > 2*M*gamma) {
        printf("M_beta: %d, M: %d\n", M_beta, M);
        FAISS_THROW_MSG("M_beta must be less than 2*M*gamma");
    }
    for (int level = 0;; level++) {
        float proba = exp(-level / levelMult) * (1 - exp(-1 / levelMult));
        if (proba < 1e-9)
            break;
        assign_probas.push_back(proba);
        nn += level == 0 ? (int) M_beta + 1.5*M : M * gamma;
        cum_nneighbor_per_level.push_back(nn);

    }
}

void ACORN::clear_neighbor_tables(int level) {
    for (int i = 0; i < levels.size(); i++) {
        size_t begin, end;
        neighbor_range(i, level, &begin, &end);
        for (size_t j = begin; j < end; j++) {
            neighbors[j] = ACORN::NeighNode(-1);
        }
    }
}

void ACORN::reset() {
    max_level = -1;
    entry_point = -1;
    offsets.clear();
    offsets.push_back(0);
    levels.clear();
    neighbors.clear();
}


void ACORN::print_neighbor_stats(int level) const {
    FAISS_THROW_IF_NOT(level < cum_nneighbor_per_level.size());
    printf("* stats on level %d, max %d neighbors per vertex:\n",
           level,
           nb_neighbors(level));

    size_t tot_neigh = 0, tot_common = 0, tot_reciprocal = 0, n_node = 0;
#pragma omp parallel for reduction(+: tot_neigh) reduction(+: tot_common) \
  reduction(+: tot_reciprocal) reduction(+: n_node)
    for (int i = 0; i < levels.size(); i++) {
        if (levels[i] > level) {
            n_node++;
            size_t begin, end;
            neighbor_range(i, level, &begin, &end);
            std::unordered_set<int> neighset;
            for (size_t j = begin; j < end; j++) {
                if (neighbors[j] < 0) // mod
                    break;
                neighset.insert(neighbors[j]); // mod
            }
            int n_neigh = neighset.size();
            int n_common = 0;
            int n_reciprocal = 0;
            for (size_t j = begin; j < end; j++) {
                storage_idx_t i2 = neighbors[j]; // mod
                if (i2 < 0)
                    break;
                size_t begin2, end2;
                neighbor_range(i2, level, &begin2, &end2);
                for (size_t j2 = begin2; j2 < end2; j2++) {
                    storage_idx_t i3 = neighbors[j2];  //mod
                    if (i3 < 0)
                        break;
                    if (i3 == i) {
                        n_reciprocal++;
                        continue;
                    }
                    if (neighset.count(i3)) {
                        neighset.erase(i3);
                        n_common++;
                    }
                }
            }
            tot_neigh += n_neigh;
            tot_common += n_common;
            tot_reciprocal += n_reciprocal;
        }
    }
    float normalizer = n_node;
    printf("   1. nb of nodes: %zd\n", n_node);
    printf("   2. neighbors per node: %.2f (%zd)\n",
           tot_neigh / normalizer,
           tot_neigh);
    printf("   3. nb of reciprocal neighbors: %.2f\n",
           tot_reciprocal / normalizer);
    printf("   4. nb of neighbors that are also neighbor-of-neighbors: %.2f (%zd)\n",
           tot_common / normalizer,
           tot_common);
}


// same as print_neighbor_stats with additional edge lists printed, no parallelism
void ACORN::print_edges(int level) const {
    FAISS_THROW_IF_NOT(level < cum_nneighbor_per_level.size());
    printf("* stats on level %d, max %d neighbors per vertex:\n",
           level,
           nb_neighbors(level));

    size_t tot_neigh = 0, tot_common = 0, tot_reciprocal = 0, n_node = 0;
    printf("\t edges lists:\n");
//#pragma omp parallel for reduction(+: tot_neigh) reduction(+: tot_common) \
  reduction(+: tot_reciprocal) reduction(+: n_node)
    for (int i = 0; i < levels.size(); i++) {
        if (levels[i] > level) {
            n_node++;
            size_t begin, end;
            neighbor_range(i, level, &begin, &end);
            std::unordered_set<int> neighset;
            printf("\t\t %d: [", i);
            for (size_t j = begin; j < end; j++) {
                if (neighbors[j] < 0) // mod
                    break;
                printf("%d(%d), ", neighbors[j], metadata[neighbors[j]]); // mod
                neighset.insert(neighbors[j]); // mod
            }
            printf("]\n");
            int n_neigh = neighset.size();
            int n_common = 0;
            int n_reciprocal = 0;
            for (size_t j = begin; j < end; j++) {
                storage_idx_t i2 = neighbors[j]; //mod
                if (i2 < 0)
                    break;
                FAISS_ASSERT(i2 != i);
                size_t begin2, end2;
                neighbor_range(i2, level, &begin2, &end2);
                for (size_t j2 = begin2; j2 < end2; j2++) {
                    storage_idx_t i3 = neighbors[j2]; // mod
                    if (i3 < 0)
                        break;
                    if (i3 == i) {
                        n_reciprocal++;
                        continue;
                    }
                    if (neighset.count(i3)) {
                        neighset.erase(i3);
                        n_common++;
                    }
                }
            }
            tot_neigh += n_neigh;
            tot_common += n_common;
            tot_reciprocal += n_reciprocal;
        }
    }
    float normalizer = n_node;

    printf("   0. level: %d\n", level);
    printf("   1. max neighbors per node: %d\n", nb_neighbors(level));
    printf("   2. nb of nodes: %zd\n", n_node);
    printf("   3. neighbors per node: %.2f (%zd)\n",
           tot_neigh / normalizer,
           tot_neigh);
    printf("   4. nb of reciprocal neighbors: %.2f\n",
           tot_reciprocal / normalizer);
    printf("   5. nb of neighbors that are also neighbor-of-neighbors: %.2f (%zd)\n",
           tot_common / normalizer,
           tot_common);
}

// same as print_neighbor_stats with additional edge lists printed according to filter
void ACORN::print_edges_filtered(int level, int filter, Operation op) const {
    FAISS_THROW_IF_NOT(level < cum_nneighbor_per_level.size());
    printf("* stats on level %d, max %d neighbors per vertex:\n",
           level,
           nb_neighbors(level));

    size_t tot_neigh = 0, tot_common = 0, tot_reciprocal = 0, n_node = 0;
    printf("\t edges lists:\n");
//#pragma omp parallel for reduction(+: tot_neigh) reduction(+: tot_common) \
  reduction(+: tot_reciprocal) reduction(+: n_node)
    for (int i = 0; i < levels.size(); i++) {
        if (levels[i] > level) {   
            if (((op == EQUAL) && (metadata[i] == filter)) || (op == OR && ((metadata[i] & filter) != 0))) {
            // if (metadata[i] == filter) {
                n_node++;
                size_t begin, end;
                neighbor_range(i, level, &begin, &end);
                std::unordered_set<int> neighset;
                printf("\t\t %d (%d): [", i, metadata[i]);

                for (size_t j = begin; j < end; j++) {
                    if (neighbors[j] < 0) // mod
                        break;
                    if (((op == EQUAL) && (metadata[neighbors[j]] == filter)) || (op == OR && ((metadata[neighbors[j]] & filter) != 0))) {
                    // if (neighbors[j].second == filter) {
                        printf("%d(%d), ", neighbors[j], metadata[neighbors[j]]); // mod
                        neighset.insert(neighbors[j]); // mod
                    }
                }
                // if (metadata[i] == filter) {
                //     printf("]\n");
                // }
                printf("]\n");


                int n_neigh = neighset.size();
                int n_common = 0;
                int n_reciprocal = 0;
                for (size_t j = begin; j < end; j++) {
                    storage_idx_t i2 = neighbors[j]; //mod
                    if (i2 < 0)
                        break;
                    FAISS_ASSERT(i2 != i);
                    size_t begin2, end2;
                    neighbor_range(i2, level, &begin2, &end2);
                    for (size_t j2 = begin2; j2 < end2; j2++) {
                        storage_idx_t i3 = neighbors[j2]; // mod
                        if (i3 < 0)
                            break;
                        if (i3 == i) {
                            n_reciprocal++;
                            continue;
                        }
                        if (neighset.count(i3)) {
                            neighset.erase(i3);
                            n_common++;
                        }
                    }
                }
                tot_neigh += n_neigh;
                tot_common += n_common;
                tot_reciprocal += n_reciprocal;
            }


            
        }
    }
    float normalizer = n_node;

    printf("   0. level: %d\n", level);
    printf("   1. max neighbors per node: %d\n", nb_neighbors(level));
    printf("   2. nb of nodes: %zd\n", n_node);
    printf("   3. neighbors per node: %.2f (%zd)\n",
           tot_neigh / normalizer,
           tot_neigh);
    printf("   4. nb of reciprocal neighbors: %.2f\n",
           tot_reciprocal / normalizer);
    printf("   5. nb of neighbors that are also neighbor-of-neighbors: %.2f (%zd)\n",
           tot_common / normalizer,
           tot_common);

}




// stats for all levels 
void ACORN::print_neighbor_stats(bool edge_list, bool filtered_edge_list, int filter, Operation op) const {
    printf("========= METADATA =======\n");

    printf("\t* cumulative max num neighbors per level\n");
    for (int i = 0; i < cum_nneighbor_per_level.size() - 1; i++) {
        printf("\t\tindx %d: %d\n", i, cum_nneighbor_per_level[i]);
    }
    printf("\t* level probabilities\n");
    for (int level = 0; level < assign_probas.size(); level++) {
        printf("\t\tlevel %d: %f\n", level, assign_probas[level]);
    }
    printf("\t* efConstruction: %d\n", efConstruction);
    printf("\t* efSearch: %d\n", efSearch);
    printf("\t* max_level: %d\n", max_level);
    printf("\t* entry_point: %d\n", entry_point);
    printf("\t* gamma: %d\n", gamma);


    // per level stats
    printf("========= LEVEL STATS OF ACORN =======\n");
    if (edge_list){
        // skip bottom level since its too big
        for (int level = 0; level <= max_level; level++) {
            printf("========= LEVEL %d =======\n", level);
            print_edges(level); // this is more detailed version with edge lists
            
        } 
    } else {
        for (int level = 0; level <= max_level; level++) {
            printf("========= LEVEL %d =======\n", level);
            print_neighbor_stats(level);
            
        }
    }

    // optional filtered edge list print
    if (filtered_edge_list) {
        printf("========= LEVEL STATS OF SUBGRAPH WITH FILTER %d, OP %d (0 for EQUAL, 1 for OR)=======\n", filter, op);
        for (int level = 0; level <= max_level; level++) {
            printf("========= LEVEL %d =======\n", level);
            print_edges_filtered(level, filter, op); // this is more detailed version with edge lists
            
        }
    } 
    // print metadata
    // printf("========= METADATA =======\n");
    // for (int i = 0; i < metadata.size(); i++) {
    //     printf("\t%d: %d\n", i, metadata[i]);
    // }
}



void ACORN::fill_with_random_links(size_t n) {
    throw FaissException("UNIMPLEMENTED");
    // int max_level = prepare_level_tab(n);
    // RandomGenerator rng2(456);

    // for (int level = max_level - 1; level >= 0; --level) {
    //     std::vector<int> elts;
    //     for (int i = 0; i < n; i++) {
    //         if (levels[i] > level) {
    //             elts.push_back(i);
    //         }
    //     }
    //     printf("linking %zd elements in level %d\n", elts.size(), level);

    //     if (elts.size() == 1)
    //         continue;

    //     for (int ii = 0; ii < elts.size(); ii++) {
    //         int i = elts[ii];
    //         size_t begin, end;
    //         neighbor_range(i, 0, &begin, &end);
    //         for (size_t j = begin; j < end; j++) {
    //             int other = 0;
    //             do {
    //                 other = elts[rng2.rand_int(elts.size())];
    //             } while (other == i);

    //             neighbors[j] = other;
    //         }
    //     }
    // }
}

// Modified from original HNSW
// n is the number of vectors that will be added (this gets called in IndexHSNW.hnsw_add_vertices)
int ACORN::prepare_level_tab(size_t n, bool preset_levels) {
    size_t n0 = offsets.size() - 1;

    if (preset_levels) {
        FAISS_ASSERT(n0 + n == levels.size());
    } else {
        FAISS_ASSERT(n0 == levels.size());
        for (int i = 0; i < n; i++) {
            int pt_level = random_level();
            levels.push_back(pt_level + 1);
        }
    }

    int max_level = 0;
    for (int i = 0; i < n; i++) {
        int pt_level = levels[i + n0] - 1;
        if (pt_level > max_level)
            max_level = pt_level;
        offsets.push_back(offsets.back() + cum_nb_neighbors(pt_level + 1));
        // neighbors.resize(offsets.back(), -1);
        neighbors.resize(offsets.back(), NeighNode(-1)); // mod
    }

    return max_level;
}

/** Enumerate vertices from farthest to nearest from query, keep a
 * neighbor only if there is no previous neighbor that is closer to
 * that vertex than the query.
 */
void ACORN::shrink_neighbor_list(
        DistanceComputer& qdis,
        std::priority_queue<NodeDistFarther>& input,
        std::vector<NodeDistFarther>& output,
        int max_size, int gamma, storage_idx_t q_id, int q_attr) {

    debug("shrink_neighbor_list: input size: %ld, max_size: %d, gamma: %d\n", input.size(), max_size, gamma);
    // new pruning method which removes neighbors are in an existing neighbors neighborhood
    std::unordered_set<storage_idx_t> neigh_of_neigh;
        // for (const NodeDistFarther& node : output) {
        //     outputSet.insert(node.id);
        // }
    int node_num = 0;    
    
    while (input.size() > 0) {
        node_num = node_num + 1;

        NodeDistFarther v1 = input.top();
        input.pop();
        float dist_v1_q = v1.d;

        debug("shrink_neighbor_list: checking whether to keep v1: %d\n", v1.id);
        bool good = true;

        // check if current candidate is in the neighbors of output 
        if (node_num > this->M_beta  && neigh_of_neigh.count(v1.id) > 0) {
            good = false;
            debug("PRUNE v1: %d\n", v1.id);
        }
        
        
        if (good) {
            output.push_back(v1);
            if (output.size() >= max_size) {
                return;
            }

            // update neigh of neigh set
            neigh_of_neigh.insert(v1.id);
            if (node_num > this->M_beta) {
                size_t begin, end;
                neighbor_range(v1.id, 0, &begin, &end);
                for (size_t j = begin; j < end; j++) {
                    if (neighbors[j] < 0) // mod
                        break;
                    neigh_of_neigh.insert(neighbors[j]); // mod
                }

            } 
            
            // break if neigh_of_neigh set is sufficiently large
            if (neigh_of_neigh.size() >= max_size) {
                break;
            }

        }
        debug("tentative edge list of size %zd: \n", output.size());
        for (NodeDistFarther v2 : output) {
            debug("\t\t node %d\n", v2.id);
        }
        debug("---- %zd: \n", output.size());
        
    }
    debug("final edge list of size %zd: \n", output.size());
        for (NodeDistFarther v2 : output) {
            debug("\t\t node %d\n", v2.id);
        }
        debug("---- %zd: \n", output.size());

}

namespace {

using storage_idx_t = ACORN::storage_idx_t;
using NodeDistCloser = ACORN::NodeDistCloser;
using NodeDistFarther = ACORN::NodeDistFarther;

/**************************************************************
 * Addition subroutines
 **************************************************************/

/// remove neighbors from the list to make it smaller than max_size
void shrink_neighbor_list(
        DistanceComputer& qdis,
        std::priority_queue<NodeDistCloser>& resultSet1,
        int max_size, int gamma, storage_idx_t q_id, int q_attr, ACORN& hnsw) {
    debug("shrink_neighbor_list from size %ld, to max size %d\n", resultSet1.size(), max_size);
    // if (resultSet1.size() < max_size) {
    //     return;
    // }
    std::priority_queue<NodeDistFarther> resultSet;
    std::vector<NodeDistFarther> returnlist;

    while (resultSet1.size() > 0) {
        resultSet.emplace(resultSet1.top().d, resultSet1.top().id);
        resultSet1.pop();
    }

    // ACORN::shrink_neighbor_list(qdis, resultSet, returnlist, max_size, gamma, q_id, q_attr);
    hnsw.shrink_neighbor_list(qdis, resultSet, returnlist, max_size, gamma, q_id, q_attr);


    for (NodeDistFarther curen2 : returnlist) {
        resultSet1.emplace(curen2.d, curen2.id);
    }
}

// modified from normal hnsw
/// add a link between two elements, possibly shrinking the list
/// of links to make room for it.
void add_link(
        ACORN& hnsw,
        DistanceComputer& qdis,
        storage_idx_t src,
        storage_idx_t dest,
        int level) {
    size_t begin, end;
    hnsw.neighbor_range(src, level, &begin, &end);
    if (hnsw.neighbors[end - 1] == -1) { // mood
        // there is enough room, find a slot to add it
        size_t i = end;
        while (i > begin) {
            if (hnsw.neighbors[i - 1] != -1) // mod
                break;
            i--;
        }
        // hnsw.neighbors[i] = dest;
        hnsw.neighbors[i] = dest; // mod
        debug("added link from %d to %d at level %d\n", src, dest, level);
        return;
    }

    // otherwise we let them fight out which to keep

    // copy to resultSet...
    std::priority_queue<NodeDistCloser> resultSet;
    resultSet.emplace(qdis.symmetric_dis(src, dest), dest);
    for (size_t i = begin; i < end; i++) { // HERE WAS THE BUG
        // storage_idx_t neigh = hnsw.neighbors[i];
        // auto [neigh, metadata] = hnsw.neighbors[i]; // mod
        auto neigh = hnsw.neighbors[i];
        auto metadata = hnsw.metadata[neigh];
        resultSet.emplace(qdis.symmetric_dis(src, neigh), neigh);
    }


    debug("calling shrink neigbor list, src: %d, dest: %d, level: %d\n", src, dest, level);
    
    if (level == 0) {
        shrink_neighbor_list(qdis, resultSet, end - begin, hnsw.gamma, src, hnsw.metadata[src], hnsw);

    }
    
    

    // ...and back
    size_t i = begin;
    while (resultSet.size()) {
        // hnsw.neighbors[i++] = resultSet.top().id;
        hnsw.neighbors[i++] = ACORN::NeighNode(resultSet.top().id); // mod
        resultSet.pop();
    }
    // they may have shrunk more than just by 1 element
    while (i < end) {
        // hnsw.neighbors[i++] = -1;
        hnsw.neighbors[i++] = ACORN::NeighNode(-1); //mod
    }
}


// modified from normal hnsw
/// search neighbors on a single level, starting from an entry point
// this only gets called in construction
void search_neighbors_to_add(
        ACORN& hnsw,
        DistanceComputer& qdis,
        std::priority_queue<NodeDistCloser>& results,
        int entry_point,
        float d_entry_point,
        int level,
        VisitedTable& vt,
        std::vector<storage_idx_t> ep_per_level = {}) {
    debug("search_neighbors to add, entrypoint: %d\n", entry_point);
    // top is nearest candidate
    std::priority_queue<NodeDistFarther> candidates;

    NodeDistFarther ev(d_entry_point, entry_point);
    candidates.push(ev);
    results.emplace(d_entry_point, entry_point);
    vt.set(entry_point);

    // number of neighbors we want for hybrid construction
    // int M = hnsw.nb_neighbors(level);
    int M;
    if (level == 0) { // level 0 may be compressed
        M = 2* hnsw.M * hnsw.gamma;
    }
    else {
        M = hnsw.nb_neighbors(level);
    }
    debug("desired resuts size: %d, at level: %d\n", M, level);
    

    int backtrack_level = level;

    while (!candidates.empty()) {
        // get nearest
        const NodeDistFarther& currEv = candidates.top();

        // MOD greedy break - added break conditional on size of results
        if ((currEv.d > results.top().d && hnsw.gamma == 1) || results.size() >= M) {
            debug("greedy stop in construction, results size: %ld, desired size: %d, gamma = %d\n", results.size(), M, hnsw.gamma);
            break;
        }
        int currNode = currEv.id;
        candidates.pop();

        // loop over neighbors
        size_t begin, end;
        hnsw.neighbor_range(currNode, level, &begin, &end);
        
        int numIters = 0;

        debug("checking neighbors of %d\n", currNode);
        for (size_t i = begin; i < end; i++) {
            // auto [nodeId, metadata] = hnsw.neighbors[i]; // storage_idx_t, int
            auto nodeId = hnsw.neighbors[i];
            auto metadata = hnsw.metadata[nodeId];
            // storage_idx_t nodeId = hnsw.neighbors[i];
            if (nodeId < 0)
                break;
            if (vt.get(nodeId))
                continue;
            vt.set(nodeId);

            // limit number neighbors visisted during construciton
            numIters = numIters + 1;
            if (numIters > hnsw.M) {
                break;
            }

            float dis = qdis(nodeId);
            NodeDistFarther evE1(dis, nodeId);

            // debug("while checking neighbors of %d, efc: %d, results size: %ld, just visited %d\n", currNode, hnsw.efConstruction, results.size(), nodeId);
            if (results.size() < hnsw.efConstruction || results.top().d > dis) {
                results.emplace(dis, nodeId);
                candidates.emplace(dis, nodeId);
                if (results.size() > hnsw.efConstruction) {
                    results.pop();
                }
            }
            debug("while checking neighbors of %d, just visited %d -- efc: %d, results size: %ld, candidates size: %ld, \n", currNode, nodeId, hnsw.efConstruction, results.size(), candidates.size());

            

            // limit number neighbors visisted during construciton
            numIters = numIters + 1;
            if (numIters > hnsw.M) {
                break;
            }

        
        }
        
        debug("during BFS, gamma: %d, candidates size: %ld, results size: %ld, vt.num_visited: %d, nb on level: %d, backtrack_level: %d, level: %d\n", hnsw.gamma, candidates.size(), results.size(), vt.num_visited(), hnsw.nb_per_level[level], backtrack_level, level);
    }
    debug("search_neighbors to add finds %ld nn's\n", results.size());
    // printf("search_neighbors to add finds %ld nn's\n", results.size());
    // printf("\tdesired resuts size: %d, at level: %d\n", M, level);

    vt.advance();
}


/**************************************************************
 * Searching subroutines
 **************************************************************/

/// greedily update a nearest vector at a given level
/// used for construction (other version below will be used in search)
void greedy_update_nearest(
        const ACORN& hnsw,
        DistanceComputer& qdis,
        int level,
        storage_idx_t& nearest,
        float& d_nearest) {
    debug("%s\n", "reached");
    for (;;) {
        storage_idx_t prev_nearest = nearest;

        size_t begin, end;
        hnsw.neighbor_range(nearest, level, &begin, &end);
        
        
        int numIters = 0;

        for (size_t i = begin; i < end; i++) {
            auto v = hnsw.neighbors[i];
            auto metadata = hnsw.metadata[v];
            if (v < 0) {
                break;
            }

            // limit number neighbors visisted during construciton
            numIters = numIters + 1;
            if (numIters > hnsw.M) {
                break;
            }

            float dis = qdis(v);
            if (dis < d_nearest) {
                nearest = v;
                d_nearest = dis;
            }
        }
        if (nearest == prev_nearest) {
            return;
        }
    }
}


/// for hybrid search only
int hybrid_greedy_update_nearest(
        const ACORN& hnsw,
        DistanceComputer& qdis,
        char* filter_map,
        // int filter,
        // Operation op,
        // std::string regex,
        int level,
        storage_idx_t& nearest,
        float& d_nearest) {
    debug("%s\n", "reached"); 
    // printf("hybrid_greedy_update_nearest called with parameters: filter: %d, op: %d, regex: %s, level: %d\n", filter, op, regex.c_str(), level);
    int ndis = 0;
    for (;;) {
        int num_found = 0;
        storage_idx_t prev_nearest = nearest;
        debug_search("----hybrid_greedy_update visists current nearest: %d, d_nearest: %f\n", nearest, d_nearest);

        size_t begin, end;
        hnsw.neighbor_range(nearest, level, &begin, &end);
        debug_search("%s", "--------checking neighbors: \n");
        
        // for debugging, collect all neighbors looked at in a vector
        std::vector<std::pair<storage_idx_t, int>> neighbors_checked;
        bool keep_expanding = true;

        for (size_t i = begin; i < end; i++) {
            auto v = hnsw.neighbors[i];
            
            if (v < 0)
                break;
                
            // note that this slows down search significantly but can be useful for debugging
            // if (debugSearchFlag) {
            //     neighbors_checked.push_back(std::make_pair(v, metadata)); 
            //     debug_search("------------checking neighbor: %d, metadata: %d, metadata & filter: %d\n", v, metadata, metadata & filter);
            // }

            // filter
            // printf("---at first filter: op: %d, metadata: %s, regex: %s, check_regex result: %d\n", op, hnsw.metadata_strings[v].c_str(), regex.c_str(), CHECK_REGEX(hnsw.metadata_strings[v], regex));
            if (filter_map[v]) {
                num_found = num_found + 1;
            } else {
                // not filter & gamma > 1
                if (hnsw.gamma > 1) {
                    continue;
                }
            }
            

        
            
            // check if filter pass
            if (filter_map[v]) {
    
                float dis = qdis(v);
                ndis += 1;
                if (dis < d_nearest || !filter_map[nearest]) {
                
                    nearest = v;
                    d_nearest = dis;
                    // debug_search("----------------new nearest: %d, d_nearest: %f\n", nearest, d_nearest);
                }
                if (num_found >= hnsw.M) {
                    // debug_search("----found %d neighbors with filter %d, returning\n", num_found, filter);
                    break;
                }
            }            

          

            // expand neighbor list if gamma=1
            if (hnsw.gamma == 1) {
                size_t begin2, end2;
                hnsw.neighbor_range(v, level, &begin2, &end2);
                for (size_t j = begin2; j < end2; j++) {
                    auto v2 = hnsw.neighbors[j];
                   

                    if (v2 < 0)
                        break;


                    // check filter pass
                    if (filter_map[v2]) {
                        num_found = num_found + 1;
                        float dis2 = qdis(v2);
                        ndis += 1;
                        // debug_search("------------found: %d, metadata: %d distance to v: %f\n", v2, metadata2, dis2);
          
                        if (dis2 < d_nearest || !filter_map[nearest]) {
                            nearest = v2;
                            d_nearest = dis2;
                            // debug_search("----------------new nearest: %d, d_nearest: %f\n", nearest, d_nearest);
                        }
                        if (num_found >= hnsw.M) {
                            break;
                        }
                    } 
                   
                }
            }
        }       

        if (nearest == prev_nearest) {
            return ndis;
        }
    }
    return ndis;
}

} // namespace

// modified from normal hnsw
void ACORN::add_links_starting_from(
        DistanceComputer& ptdis,
        storage_idx_t pt_id,
        storage_idx_t nearest,
        float d_nearest,
        int level,
        omp_lock_t* locks,
        VisitedTable& vt,
        std::vector<storage_idx_t> ep_per_level) {
    debug("add_links_starting_from at level: %d, nearest: %d\n", level, nearest);
    std::priority_queue<NodeDistCloser> link_targets;

    search_neighbors_to_add(
            *this, ptdis, link_targets, nearest, d_nearest, level, vt, ep_per_level); // mod

    // added update nearest
    nearest = link_targets.top().id;

    // but we can afford only this many neighbors
    int M = nb_neighbors(level);

    debug("add_links_starting_from will shrink results list to size: %d\n", M);

    
    debug("calling shrink neigbor list, pt_id: %d, level: %d\n", pt_id, level);

    if (level == 0) {
        ::faiss::shrink_neighbor_list(ptdis, link_targets, M, gamma, pt_id, this->metadata[pt_id], *this);
        // printf("shrunk");
    }
    
    
    debug("add_links_starting_from gets edge link size: %ld\n", link_targets.size());

    std::vector<storage_idx_t> neighbors;
    neighbors.reserve(link_targets.size());
    while (!link_targets.empty()) {
        storage_idx_t other_id = link_targets.top().id;
        add_link(*this, ptdis, pt_id, other_id, level); // mod
        neighbors.push_back(other_id);
        link_targets.pop();
    }

    omp_unset_lock(&locks[pt_id]);
    for (storage_idx_t other_id : neighbors) {
        omp_set_lock(&locks[other_id]);
        add_link(*this, ptdis, other_id, pt_id, level); // mod
        omp_unset_lock(&locks[other_id]);
    }
    omp_set_lock(&locks[pt_id]);
}



/**************************************************************
 * Building, parallel
 **************************************************************/
// mod compared to original hnsnw
void ACORN::add_with_locks(
        DistanceComputer& ptdis,
        int pt_level,
        int pt_id,
        std::vector<omp_lock_t>& locks,
        VisitedTable& vt) {
    //  greedy search on upper levels
    debug("add_with_locks called for node %d, level %d\n", pt_id, pt_level);


    storage_idx_t nearest;
#pragma omp critical
    {
        nearest = entry_point;

        if (nearest == -1) {
            max_level = pt_level;
            entry_point = pt_id;
            for (int i=0; i <= max_level; i++){
                nb_per_level[i] = nb_per_level[i] + 1;
            }
            
        }
    }

    if (nearest < 0) {
        return;
    }

    omp_set_lock(&locks[pt_id]);

    int level = max_level; // level at which we start adding neighbors
    float d_nearest = ptdis(nearest);

    // needed for backtracking in hybrid search - TODO DEPRECATED, remove backtracking things
    std::vector<storage_idx_t> ep_per_level(max_level); // idx of nearest node per level
    ep_per_level[level] = nearest;

    for (; level > pt_level; level--) {
        debug("--greedy update nearest at level: %d, ep: %d\n", level, nearest);
        greedy_update_nearest(*this, ptdis, level, nearest, d_nearest);
        ep_per_level[level] = nearest;
    }

    for (int i = 0; i <= max_level; i++) {
        debug("--at level: %d, ep: %d\n", i, ep_per_level[i]);
    }

    for (; level >= 0; level--) {
        debug("--add_links_starting_from at level: %d\n", level);
        add_links_starting_from(
                ptdis, pt_id, nearest, d_nearest, level, locks.data(), vt, ep_per_level);
        nb_per_level[level] = nb_per_level[level] + 1;
    }

    omp_unset_lock(&locks[pt_id]);

    if (pt_level > max_level) {
        max_level = pt_level;
        entry_point = pt_id;
    }
}

/**************************************************************
 * Searching
 **************************************************************/

namespace {

using MinimaxHeap = ACORN::MinimaxHeap;
using Node = ACORN::Node;
using NeighNode = ACORN::NeighNode;
/** Do a BFS on the candidates list */
// this is called in search and search_from_level_0
int search_from_candidates(
        const ACORN& hnsw,
        DistanceComputer& qdis,
        int k,
        idx_t* I,
        float* D,
        MinimaxHeap& candidates,
        VisitedTable& vt,
        ACORNStats& stats,
        int level,
        int nres_in = 0,
        const SearchParametersACORN* params = nullptr) {
    debug("%s\n", "reached");
    int nres = nres_in;
    int ndis = 0;

    // can be overridden by search params
    bool do_dis_check = params ? params->check_relative_distance
                               : hnsw.check_relative_distance;
    int efSearch = params ? params->efSearch : hnsw.efSearch;
    const IDSelector* sel = params ? params->sel : nullptr;

    for (int i = 0; i < candidates.size(); i++) {
        idx_t v1 = candidates.ids[i];
        float d = candidates.dis[i];
        FAISS_ASSERT(v1 >= 0);
        if (!sel || sel->is_member(v1)) {
            if (nres < k) {
                faiss::maxheap_push(++nres, D, I, d, v1);
            } else if (d < D[0]) {
                faiss::maxheap_replace_top(nres, D, I, d, v1);
            }
        }
        vt.set(v1);
    }

    int nstep = 0;

    while (candidates.size() > 0) { // candidates is heap of size max(efs, k)
        float d0 = 0;
        int v0 = candidates.pop_min(&d0);

        if (do_dis_check) {
            // tricky stopping condition: there are more that ef
            // distances that are processed already that are smaller
            // than d0

            int n_dis_below = candidates.count_below(d0);
            if (n_dis_below >= efSearch) {
                break;
            }
        }

        size_t begin, end;
        hnsw.neighbor_range(v0, level, &begin, &end);

        for (size_t j = begin; j < end; j++) {
            int v1 = hnsw.neighbors[j];
            if (v1 < 0)
                break;
            if (vt.get(v1)) {
                continue;
            }
            vt.set(v1);
            ndis++;
            float d = qdis(v1);
            if (!sel || sel->is_member(v1)) {
                if (nres < k) {
                    faiss::maxheap_push(++nres, D, I, d, v1);
                } else if (d < D[0]) {
                    faiss::maxheap_replace_top(nres, D, I, d, v1);
                }
            }
            candidates.push(v1, d);
        }

        nstep++;
        if (!do_dis_check && nstep > efSearch) {
            break;
        }
    }

    if (level == 0) {
        stats.n1++;
        if (candidates.size() == 0) {
            stats.n2++;
        }
        stats.n3 += ndis;
    }

    return nres;
}

// has a filter arg for hybrid search, this only gets called on level 0
int hybrid_search_from_candidates(
        const ACORN& hnsw,
        DistanceComputer& qdis,
        char* filter_map,
        // int filter,
        // Operation op,
        // std::string regex,
        int k,
        idx_t* I,
        float* D,
        MinimaxHeap& candidates,
        VisitedTable& vt,
        ACORNStats& stats,
        int level,
        int nres_in = 0,
        const SearchParametersACORN* params = nullptr) {
    // debug("%s\n", "reached");
    // printf("----hybrid_search_from_candidates called with filter: %d, k: %d, op: %d, regex: %s\n", filter, k, op, regex.c_str());
    // debug_search("----hybrid_search_from_candidates called with filter: %d, k: %d\n", filter, k);
    int nres = nres_in;
    int ndis = 0;

    // can be overridden by search params
    bool do_dis_check = params ? params->check_relative_distance
                               : hnsw.check_relative_distance;
    int efSearch = params ? params->efSearch : hnsw.efSearch;
    const IDSelector* sel = params ? params->sel : nullptr;

    for (int i = 0; i < candidates.size(); i++) {
        idx_t v1 = candidates.ids[i];
        float d = candidates.dis[i];
        FAISS_ASSERT(v1 >= 0);
        if (!sel || sel->is_member(v1)) {
            if (nres < k) {
                faiss::maxheap_push(++nres, D, I, d, v1);
            } else if (d < D[0]) {
                faiss::maxheap_replace_top(nres, D, I, d, v1);
            }
        }
        vt.set(v1);
    }

    int nstep = 0;


    // timing variables
    double t1_candidates_loop = elapsed();
    
    while (candidates.size() > 0) { // candidates is heap of size max(efs, k)
        float d0 = 0;
        int v0 = candidates.pop_min(&d0);
        // debug_search("--------visiting v0: %d, d0: %f, candidates_size: %d\n", v0, d0, candidates.size());

        if (do_dis_check) {
            // tricky stopping condition: there are more that ef
            // distances that are processed already that are smaller
            // than d0
            int n_dis_below = candidates.count_below(d0);
            if (n_dis_below >= efSearch) {
                // debug("--------%s\n", "n_dis_below >= efSearch BREAK cond reached");
                // debug_search("--------n_dis_below: %d, efSearch: %d - triggers break\n", n_dis_below, efSearch);
                break;
            }
        }

        size_t begin, end;
        hnsw.neighbor_range(v0, level, &begin, &end);

        // variable to keep track of search expansion
        int num_found = 0;
        int num_new = 0;
        bool keep_expanding = true;

        // for debugging, collect all neighbors looked at in a vector
        std::vector<std::pair<storage_idx_t, int>> neighbors_checked;

        double t1_neighbors_loop = elapsed();
        for (size_t j = begin; j < end; j++) {
            // auto [v1, metadata] = hnsw.neighbors[j];
            bool promising = 0;
            bool outerskip = false;

            auto v1 = hnsw.neighbors[j];
            // auto metadata = hnsw.metadata[v1];
            // debug_search("------------visiting neighbor (%ld) - %d, metadata: %d\n", j-begin, v1, metadata);


            if (v1 < 0) {
                break;
            }

            // note that this slows down search performance significantly
            // if (debugSearchFlag) {
            //     neighbors_checked.push_back(std::make_pair(v1, metadata)); // for debugging
            // }
            if (filter_map[v1]) {
               num_found = num_found + 1; // increment num found
            }
            
            if (vt.get(v1)) {
                continue;
            }


            // filter
            if (filter_map[v1]) {
                vt.set(v1);
                num_new = num_new + 1; // increment num new
                ndis++;
                float d = qdis(v1);
                // debug_search("------------new candidate %d, distance: %f\n", v1, d);

                if (!sel || sel->is_member(v1)) {
                    if (nres < k) {
                        // debug_search("-----------------pushing new candidate, nres: %d (to be incrd)\n", nres);
                        faiss::maxheap_push(++nres, D, I, d, v1);
                        // debug_search("-----------------pushed new candidate, nres: %d\n", nres);
                        promising = 1;
                    } else if (d < D[0]) {
                        // debug_search("-----------------replacing top, nres: %d\n", nres);
                        faiss::maxheap_replace_top(nres, D, I, d, v1);
                        promising =1;
                    }
                }
                candidates.push(v1, d);

                if (num_found >= hnsw.M * 2) {
                    // debug_search("------------num_found: %d, M: %d - triggered outer brea, skpping to M_beta=%d neighbork\n", num_found, hnsw.M * 2, hnsw.M_beta);
                    keep_expanding = false;
                    break;
                }
            }    
            
            if (((j - begin >= hnsw.M_beta) && keep_expanding) || hnsw.gamma == 1) {
                debug_search("------------expanding neighbor list for %d; neighbor %ld, hnsw.M_beta: %d\n", v1, j-begin, hnsw.M_beta);
                size_t begin2, end2;
                hnsw.neighbor_range(v1, level, &begin2, &end2);
                // try to parallelize neighbor expansion
                for (size_t j2 = begin2; j2 < end2; j2+=1) {
                    
                    auto v2 = hnsw.neighbors[j2];

                    // note that this slows down search performance significantly when flag is on
                    // if (debugSearchFlag) {
                    //     neighbors_checked.push_back(std::make_pair(v2, metadata2)); // for debugging
                    // }
                    if (v2 < 0) {
                        // continue;
                        break;
                    }

                    // if (metadata2 == filter) {
                    if (filter_map[v2]) {
                        num_found = num_found + 1; // increment num found
                    } else {
                        continue;
                    }

        

                    if (vt.get(v2)) {
                        continue;
                    }
                    
                    vt.set(v2);
                    ndis++;
  
                    float d2 = qdis(v2);
                    // debug_search("------------new candidate from expansion %d, distance: %f\n", v2, d2);
                    if (!sel || sel->is_member(v2)) {
                        if (nres < k) {
                            // debug_search("-----------------pushing new candidate, nres: %d (to be incrd)\n", nres);
                            faiss::maxheap_push(++nres, D, I, d2, v2);
                            // debug_search("-----------------pushed new candidate, nres: %d\n", nres);

                        } else if (d2 < D[0]) {
                            // debug_search("-----------------replacing top, nres: %d\n", nres);
                            faiss::maxheap_replace_top(nres, D, I, d2, v2);
                        }
                    }
                    candidates.push(v2, d2);
                    if (num_found >= hnsw.M * 2) {
    
                        // debug_search("------------num_found: %d, 2M: %d - triggers break\n", num_found, hnsw.M * 2);
                        keep_expanding = false;
                        break;
                    }
                }


    
            }
        
            
        }

     
        

        nstep++; 
        if (!do_dis_check && nstep > efSearch) {
            break;
        }
    }

    if (level == 0) {
        stats.n1++;
        if (candidates.size() == 0) {
            stats.n2++;
        }
        stats.n3 += ndis;
    }


    return nres;
}








} // anonymous namespace

ACORNStats ACORN::search(
        DistanceComputer& qdis,
        int k,
        idx_t* I,
        float* D,
        VisitedTable& vt,
        const SearchParametersACORN* params) const {
    debug("%s\n", "reached");
    ACORNStats stats;
    if (entry_point == -1) {
        return stats;
    }
    if (upper_beam == 1) { // common branch
        debug("%s\n", "reached upper beam == 1");

        //  greedy search on upper levels
        storage_idx_t nearest = entry_point;
        float d_nearest = qdis(nearest);

        for (int level = max_level; level >= 1; level--) {
            greedy_update_nearest(*this, qdis, level, nearest, d_nearest);
        }

        // honor the search-time efSearch override when sizing the candidate
        // heap; previously only the member value (set at construction) was
        // used, so params->efSearch had no effect above the default
        int efSearch_eff = params ? params->efSearch : efSearch;
        int ef = std::max(efSearch_eff, k);
        if (search_bounded_queue) { // this is the most common branch
            debug("%s\n", "reached search bounded queue");

            MinimaxHeap candidates(ef);

            candidates.push(nearest, d_nearest);

            search_from_candidates(
                    *this, qdis, k, I, D, candidates, vt, stats, 0, 0, params);
        } else {
            debug("%s\n", "reached search_bounded_queue == False");
            throw FaissException("UNIMPLEMENTED search unbounded queue");
            
        }

        vt.advance();

    } else {
        debug("%s\n", "reached upper beam != 1");

        int candidates_size = upper_beam;
        MinimaxHeap candidates(candidates_size);

        std::vector<idx_t> I_to_next(candidates_size);
        std::vector<float> D_to_next(candidates_size);

        int nres = 1;
        I_to_next[0] = entry_point;
        D_to_next[0] = qdis(entry_point);

        for (int level = max_level; level >= 0; level--) {
            // copy I, D -> candidates

            candidates.clear();

            for (int i = 0; i < nres; i++) {
                candidates.push(I_to_next[i], D_to_next[i]);
            }

            if (level == 0) {
                nres = search_from_candidates(
                        *this, qdis, k, I, D, candidates, vt, stats, 0);
            } else {
                nres = search_from_candidates(
                        *this,
                        qdis,
                        candidates_size,
                        I_to_next.data(),
                        D_to_next.data(),
                        candidates,
                        vt,
                        stats,
                        level);
            }
            vt.advance();
        }
    }

    return stats;
}


// hybrid search TODO
ACORNStats ACORN::hybrid_search(
        DistanceComputer& qdis,
        int k,
        idx_t* I,
        float* D,
        VisitedTable& vt,
        char* filter_map,
        // int filter,
        // Operation op,
        // std::string regex,
        const SearchParametersACORN* params) const {
    debug("%s\n", "reached");
    // debug_search("Hybrid Search, params -- k: %d, filter: %d\n", k, filter);
    ACORNStats stats;
    if (entry_point == -1) {
        return stats;
    }


    if (upper_beam == 1) { // common branch
        debug("%s\n", "reached upper beam == 1");

        //  greedy search on upper levels
        storage_idx_t nearest = entry_point;
        float d_nearest = qdis(nearest);

        debug_search("-starting at ep: %d, d: %f, metadata: %d\n", nearest, d_nearest, metadata[nearest]);

        int ndis_upper = 0;
        for (int level = max_level; level >= 1; level--) {
            debug_search("-at level %d, searching for greedy nearest from current nearest: %d, dist: %f, metadata: %d\n", level, nearest, d_nearest, metadata[nearest]);
            ndis_upper += hybrid_greedy_update_nearest(*this, qdis, filter_map, level, nearest, d_nearest);
            // ndis_upper += hybrid_greedy_update_nearest(*this, qdis, filter, op, regex, level, nearest, d_nearest);
            debug_search("-at level %d, new nearest: %d, d: %f, metadata: %d\n", level, nearest, d_nearest, metadata[nearest]);
            

        }
        stats.n3 += ndis_upper;

        // honor the search-time efSearch override when sizing the candidate
        // heap; previously only the member value (set at construction) was
        // used, so params->efSearch had no effect above the default
        int efSearch_eff = params ? params->efSearch : efSearch;
        int ef = std::max(efSearch_eff, k);
        if (search_bounded_queue) { // this is the most common branch
            debug("%s\n", "reached search bounded queue");

            MinimaxHeap candidates(ef);

            candidates.push(nearest, d_nearest);
            debug_search("-starting BFS at level 0 with ef: %d, nearest: %d, d: %f, metadata: %d\n", ef, nearest, d_nearest, metadata[nearest]);
            hybrid_search_from_candidates(
                    *this, qdis, filter_map, k, I, D, candidates, vt, stats, 0, 0, params);
            

        } else {
            // TODO
            printf("UNIMPLEMENTED BRANCH for hybid search\n");
            debug("%s\n", "reached search_bounded_queue == False");
            throw FaissException("UNIMPLEMENTED search unbounded queue");

            
        }

        vt.advance();

    } else {
        debug("%s\n", "reached upper beam != 1");

        int candidates_size = upper_beam;
        MinimaxHeap candidates(candidates_size);

        std::vector<idx_t> I_to_next(candidates_size);
        std::vector<float> D_to_next(candidates_size);

        int nres = 1;
        I_to_next[0] = entry_point;
        D_to_next[0] = qdis(entry_point);

        for (int level = max_level; level >= 0; level--) {
            // copy I, D -> candidates

            candidates.clear();

            for (int i = 0; i < nres; i++) {
                candidates.push(I_to_next[i], D_to_next[i]);
            }

            if (level == 0) {
                nres = hybrid_search_from_candidates(
                        *this, qdis, filter_map, k, I, D, candidates, vt, stats, 0);
            
                
            } else {
                nres = hybrid_search_from_candidates(
                        *this,
                        qdis,
                        filter_map,
                        // filter,
                        // op,
                        // regex,
                        candidates_size,
                        I_to_next.data(),
                        D_to_next.data(),
                        candidates,
                        vt,
                        stats,
                        level);
            }
            vt.advance();
        }
    }

    return stats;
}






/**************************************************************
 * MinimaxHeap
 **************************************************************/

void ACORN::MinimaxHeap::push(storage_idx_t i, float v) {
    if (k == n) {
        if (v >= dis[0])
            return;
        faiss::heap_pop<HC>(k--, dis.data(), ids.data());
        --nvalid;
    }
    faiss::heap_push<HC>(++k, dis.data(), ids.data(), v, i);
    ++nvalid;
}

float ACORN::MinimaxHeap::max() const {
    return dis[0];
}

int ACORN::MinimaxHeap::size() const {
    return nvalid;
}

void ACORN::MinimaxHeap::clear() {
    nvalid = k = 0;
}

int ACORN::MinimaxHeap::pop_min(float* vmin_out) {
    assert(k > 0);
    // returns min. This is an O(n) operation
    int i = k - 1;
    while (i >= 0) {
        if (ids[i] != -1)
            break;
        i--;
    }
    if (i == -1)
        return -1;
    int imin = i;
    float vmin = dis[i];
    i--;
    while (i >= 0) {
        if (ids[i] != -1 && dis[i] < vmin) {
            vmin = dis[i];
            imin = i;
        }
        i--;
    }
    if (vmin_out)
        *vmin_out = vmin;
    int ret = ids[imin];
    ids[imin] = -1;
    --nvalid;

    return ret;
}

int ACORN::MinimaxHeap::count_below(float thresh) {
    int n_below = 0;
    for (int i = 0; i < k; i++) {
        if (dis[i] < thresh) {
            n_below++;
        }
    }

    return n_below;
}

} // namespace faiss
