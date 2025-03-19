#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <cstring>
#include <sys/time.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexHNSW.h>
#include <faiss/IndexACORN.h>
#include <faiss/index_io.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <arpa/inet.h>
#include <pthread.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <iosfwd>
#include <faiss/impl/platform_macros.h>
#include <cassert>
#include <thread>
#include <set>
#include <numeric>
#include <nlohmann/json.hpp>
#include "utils.cpp"

// Additional includes
#include <getopt.h>

// Custom classes (SingleLabel and MultiLabel)

// Configuration structure
struct Config {
    size_t N;                       // Number of vectors
    size_t nq;                      // Number of queries
    std::string dataset_path;       // Path to the dataset
    std::string base_file;          // Path to the base vectors
    std::string query_file;         // Path to the query vectors
    std::string ground_truth_file;  // Path to the ground truth file
    int efc = 100;                   // HNSW construction parameter
    int efs = 100;                   // HNSW search parameter
    int k = 10;                     // Number of nearest neighbors
    int d = 384;                 // Dimension of vectors (to be overwritten)
    int M = 16;                     // HNSW parameter M
    int M_beta = 16;                // Compression parameter
    int gamma = 12;                  // ACORN parameter
    std::string assignment_type = "rand";
    int alpha = 0;
    std::string dataset_name = "default_dataset"; // Dataset name
    int n_centroids;                // Number of centroids (gamma)
    int searchOnly = 0;             // Search only mode
};

// Function prototypes
void parseArguments(int argc, char* argv[], Config& config);
void loadMetadata(const Config& config, std::vector<int>& metadata, std::vector<std::string>& metadata_strings);
size_t loadQueryData(const Config& config, float*& xq, std::vector<int>& aq, std::vector<std::string>& aq_strings);
void loadGroundTruth(const Config& config, const std::vector<faiss::idx_t>& nns2);
void addVectorsToIndexes(const Config& config,  faiss::IndexACORNFlat* hybrid_index_ACORNFlat);
void writeIndexesToFiles(const Config& config,  faiss::IndexACORNFlat* hybrid_index_ACORNFlat);
void printIndexStats(   faiss::IndexACORNFlat* hybrid_index_ACORNFlat);
void searchIndexes(const Config& config,   faiss::IndexACORNFlat* hybrid_index_ACORNFlat, float* xq, std::vector<std::string>& aq_strings, const std::vector<std::string>& metadata_strings);
void prepareFilterIdsMap(const Config& config, const std::vector<std::string>& aq_strings, const std::vector<std::string>& metadata_strings, std::vector<char>& filter_ids_map);
faiss::Index* readIndexesfromFiles(const Config& config);
// Main function
int main(int argc, char* argv[]) {
    int max_threads = omp_get_max_threads();
    std::cout << "Max threads available: " << max_threads << std::endl;   
    double t0 = elapsed();

    Config config;
    parseArguments(argc, argv, config);

    std::vector<int> metadata;
    std::vector<std::string> metadata_strings;
    loadMetadata(config, metadata, metadata_strings);

    float* xq;
    std::vector<int> aq;
    std::vector<std::string> aq_strings;
    config.d = loadQueryData(config, xq, aq, aq_strings);


    
    // ACORN-gamma
    faiss::IndexACORNFlat index_ACORNFlat(config.d, config.M, config.gamma, metadata, config.M*2, faiss::METRIC_L2);
    faiss::IndexACORNFlat *hybrid_index_ACORNFlat = &index_ACORNFlat;
    hybrid_index_ACORNFlat->acorn.efSearch = config.efs; // default is 16 HybridHNSW.capp
    hybrid_index_ACORNFlat->acorn.efConstruction = config.efc; // default is 100 HybridHNSW.capp
    debug("ACORN index created%s\n", "");


    // ACORN-1
    faiss::IndexACORNFlat hybrid_index_gamma1(config.d, config.M, 1, metadata, config.M*2);
    hybrid_index_gamma1.acorn.efSearch = config.efs; // default is 16 HybridHNSW.capp

    std::cout << "SearchOnly " <<  config.searchOnly << std::endl;
    if(config.searchOnly >= 1){
        faiss::Index *index  = readIndexesfromFiles(config);
        hybrid_index_ACORNFlat = dynamic_cast<faiss::IndexACORNFlat*>(index);
        hybrid_index_ACORNFlat->acorn.efSearch = config.searchOnly; 

    }else{
        addVectorsToIndexes(config, hybrid_index_ACORNFlat);

       writeIndexesToFiles(config, hybrid_index_ACORNFlat);
    }
    printIndexStats( hybrid_index_ACORNFlat);

    searchIndexes(config, hybrid_index_ACORNFlat, xq, aq_strings, metadata_strings);


  

    printf("[%.3f s] -----DONE-----\n", elapsed() - t0);

    delete[] xq;
    return 0;
}

// Function implementations
void parseArguments(int argc, char* argv[], Config& config) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <number of vectors N> <number of queries nq> <dataset path>\n", argv[0]);
        exit(1);
    }
    config.N = strtoul(argv[1], NULL, 10);
    config.nq = strtoul(argv[2], NULL, 10);
    config.dataset_path = argv[3];
    config.base_file= argv[4];
    config.query_file= argv[5];
    config.ground_truth_file= argv[6];
    config.efc = atoi(argv[7]);
    config.M = atoi(argv[8]);
    config.M_beta = config.M;
    config.searchOnly = atoi(argv[9]);

     // Optional: parse more arguments for other configurations
    // For example:
    // config.gamma = atoi(argv[4]);
    // config.M = atoi(argv[5]);
    // config.M_beta = atoi(argv[6]);

    config.n_centroids = config.gamma;
    std::cout << "[Config] N: " << config.N << ", nq: " << config.nq << ", dataset_path: " << config.dataset_path << std::endl;
}

void loadMetadata(const Config& config, std::vector<int>& metadata, std::vector<std::string>& metadata_strings) {
    double t0 = elapsed();
    // Placeholder for loading metadata
    // Implement actual data loading based on your dataset format
    // For now, we simulate loading metadata

    // fake the orginal metadata since it will have segment fault
    metadata = std::vector<int>(config.N*config.nq, 0);
    metadata_strings = load_metadata_strings(config.dataset_path + "/base_labels.txt", config.N);

    //assert(config.N == metadata.size());
    printf("[%.3f s] Loaded metadata, %ld attributes found\n", elapsed() - t0, metadata.size());
}

size_t loadQueryData(const Config& config, float*& xq, std::vector<int>& aq, std::vector<std::string>& aq_strings) {
    double t0 = elapsed();
    printf("[%.3f s] Loading query vectors and attributes\n", elapsed() - t0);

    size_t d2;
    size_t nq_loaded;
    // Placeholder for loading query vectors
    // Implement actual data loading based on your dataset format

    // Example: Load from files (to be replaced with actual file paths)

    xq = fbin_read((config.dataset_path + "/query_dense.fbin").c_str(), &d2, &nq_loaded);

    printf("[%.3f s] Loaded query vectors from %s\n", elapsed() - t0, (config.dataset_path + "/query.fbin").c_str());
    aq = load_aq(config.dataset_name, config.n_centroids, config.alpha, config.N);
    aq_strings = load_metadata_strings(config.dataset_path + "/query_dense_labels.txt", nq_loaded);

    // Adjust nq if necessary
    if (config.nq > nq_loaded) {
        printf("Warning: nq is greater than the number of loaded queries, setting nq to %ld\n", nq_loaded);
    }
    //aq.resize(config.nq);
    //aq_strings.resize(config.nq);

    printf("[%.3f s] Loaded %ld queries\n", elapsed() - t0, config.nq);

    return d2;
}

void loadGroundTruth(const Config& config, const std::vector<faiss::idx_t>& nns2) {
    size_t nq = config.nq;
    size_t k = config.k;
    
    const std::string& gt_filename = config.dataset_path + config.ground_truth_file;
    // Open the ground truth file
    std::cout << "GT Filename: " << gt_filename << std::endl;
    std::ifstream reader(gt_filename);
    if (!reader.is_open()) {
        std::cerr << "Error opening ground truth file " << gt_filename << std::endl;
        return  ;
    }

    size_t npts, ndims;
    // Read header (number of points and dimensions)
    reader >> npts >> ndims;
    reader.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); // Skip to the next line

    if (npts != nq) {
        std::cerr << "Mismatch between number of queries in ground truth file and nns2" << std::endl;
     }

    size_t gt_k = std::min(k, ndims);

    // Read the ground truth IDs
    std::vector<std::unordered_set<int32_t>> gt_sets(nq);
    std::string line;
    for (size_t i = 0; i < nq; ++i) {
        std::getline(reader, line);
        std::istringstream iss(line);
        int32_t id;
        size_t count = 0;
        // Read up to gt_k IDs per query
        while (iss >> id && count < gt_k) {
            gt_sets[i].insert(id);
            count++;
        }
    }

    // Skip the distances
    reader.close();

    // Compute recall
    size_t total_correct = 0;
    for (size_t i = 0; i < nq; ++i) {
        size_t correct = 0;
        const auto& gt_set = gt_sets[i];
        for (size_t j = 0; j < k; ++j) {
            faiss::idx_t ann_id = nns2[j + i * k];
            if (gt_set.find(ann_id) != gt_set.end()) {
                correct++;
            }
        }
        total_correct += correct;
    }
    double recall = static_cast<double>(total_correct) / (nq * k);
    
    printf("Recall@%ld: %.3f\n", k, recall);
}
    
void addVectorsToIndexes(const Config& config,  faiss::IndexACORNFlat* hybrid_index_ACORNFlat) {
    double t0 = elapsed();
    printf("[%.3f s] Loading database vectors\n", elapsed() - t0);

    size_t nb, d2;
    // Placeholder for loading base vectors
    float* xb = fbin_read((config.dataset_path + "/base.fbin").c_str(), &d2, &nb);

    if (config.d != d2) {
        printf("Error: base vectors have different dimensionality (%ld) than query vectors (%d)\n", d2, config.d);
    }

    printf("[%.3f s] Loaded base vectors from %s\n", elapsed() - t0, (config.dataset_path + "/base.fbin").c_str());

    printf("[%.3f s] Adding vectors to indexes\n", elapsed() - t0);
     hybrid_index_ACORNFlat->add(config.N, xb);

    //delete[] xb;

    printf("[%.3f s] Added %ld vectors to indexes\n", elapsed() - t0, config.N);
}

void writeIndexesToFiles(const Config& config,   faiss::IndexACORNFlat* hybrid_index_ACORNFlat) {
    double t0 = elapsed();
    std::cout << "====================Write Index====================\n" << std::endl;

    std::stringstream filepath_stream;
    filepath_stream << config.dataset_path << "/hybrid_index_M=" << config.M << "_efc=" << config.efc << "_Mb=" << config.M_beta << "_gamma=" << config.gamma <<"_N"<< config.N<<"_M"<<config.M << ".index";
    std::string hybrid_index_path = filepath_stream.str();

    write_index(hybrid_index_ACORNFlat, hybrid_index_path.c_str());
    printf("[%.3f s] Wrote hybrid index to file: %s\n", elapsed() - t0, hybrid_index_path.c_str());

 
}

faiss::Index* readIndexesfromFiles(const Config& config) {
    double t0 = elapsed();
    std::cout << "====================Read Index====================\n" << std::endl;

    std::stringstream filepath_stream;
    filepath_stream << config.dataset_path << "/hybrid_index_M=" << config.M << "_efc=" << config.efc << "_Mb=" << config.M_beta << "_gamma=" << config.gamma <<"_N"<< config.N<<"_M"<<config.M << ".index";
    std::string hybrid_index_path = filepath_stream.str();

    printf("[%.3f s] Read hybrid index from file: %s\n", elapsed() - t0, hybrid_index_path.c_str());
    faiss::Index * index  = faiss::read_index(hybrid_index_path.c_str());

    return index;
}

void printIndexStats(  faiss::IndexACORNFlat* hybrid_index_ACORNFlat) {
 
    printf("====================================\n");
    printf("============ ACORN INDEX =============\n");
    printf("====================================\n");
    hybrid_index_ACORNFlat->printStats(false);
}

void searchIndexes(const Config& config, faiss::IndexACORNFlat* hybrid_index_ACORNFlat, float* xq, std::vector<std::string>& aq_strings, const std::vector<std::string>& metadata_strings) {
    double t0 = elapsed();
    printf("==============================================\n");
    printf("====================Search Results====================\n");
    printf("==============================================\n");


    int nq_print = std::min(5, (int)config.nq);
   

    // Searching the hybrid index
    printf("==================== ACORN INDEX ====================\n");
    printf("[%.3f s] Searching the %d nearest neighbors of %ld vectors in the index, efsearch %d\n",
           elapsed() - t0, config.k, config.nq, hybrid_index_ACORNFlat->acorn.efSearch);

    std::vector<faiss::idx_t> nns2(config.k * config.nq);
    std::vector<float> dis2(config.k * config.nq);

    printf("[%.3f s] *** Start filter_ids_map %.3f\n", elapsed() - t0);



    std::vector<char> filter_ids_map(config.nq * config.N);
    std::vector<MultiLabel> query_labels(config.nq);
    std::vector<MultiLabel> base_labels(config.N);

    #pragma omp parallel for schedule(dynamic, 128)
    for (size_t xq_idx = 0; xq_idx < config.nq; xq_idx++) {
        query_labels[xq_idx] = MultiLabel::fromQuery(aq_strings[xq_idx]);
    }

    #pragma omp parallel for schedule(dynamic, 128)
    for (size_t xb_idx = 0; xb_idx < config.N; xb_idx++) {
        base_labels[xb_idx] = MultiLabel::fromBase(metadata_strings[xb_idx]);
    }

    // Prepare filter_ids_map
    double t1_f = elapsed();
    //#pragma omp parallel for schedule(dynamic, 128)
    for (size_t xq_idx = 0; xq_idx < config.nq; xq_idx++) {
        for (size_t xb_idx = 0; xb_idx < config.N; xb_idx++) {
             
            filter_ids_map[xq_idx * config.N + xb_idx] =  query_labels[xq_idx].isSubsetOf(base_labels[xb_idx]);
        }
    }
    double t2_f = elapsed();

    printf("[%.3f s] *** Done filter_ids_map %.3f\n", elapsed() - t0, t2_f - t1_f);

    double t3 = elapsed();
    hybrid_index_ACORNFlat->search(config.nq, xq, config.k, dis2.data(), nns2.data(), filter_ids_map.data());

    double t4 = elapsed();

    printf("[%.3f s] *** Done Query %.3f\n", elapsed() - t0, t4 - t3);

   
    // print config.nq, xq, config.k, dis2.data(), nns2.data(), filter_ids_map.data()
    std::cout << "config nq: " << config.nq <<" dis size:"<< dis2.size() << "nns size:"<< nns2.size() <<"id map:"<< filter_ids_map.size() << std::endl;

    std::cout << "aq_strings size: " << aq_strings.size() << std::endl;
    std::cout<< "metadata_strings size: " << metadata_strings.size() << std::endl;


    printf("[%.3f s] Query results (vector ids, then distances):\n", elapsed() - t0);
    for (int i = 0; i < nq_print; i++) {
        printf("query %2d nn's (%s): ", i, aq_strings[i].c_str());
        for (int j = 0; j < config.k; j++) {
            int id = nns2[j + i * config.k];
            if (id < 0 || id >= metadata_strings.size() ){
                printf("%d (Not Found) ", id);
            }else{
            printf("%7ld (%s) ", id, metadata_strings[id].c_str());
            }
        }
        printf("\n     dis: \t");
        for (int j = 0; j < config.k; j++) {
            printf("%7g ", dis2[j + i * config.k]);
        }
        printf("\n");
    }
    

    loadGroundTruth(config, nns2);
}

void prepareFilterIdsMap(
        const Config& config,
        const std::vector<std::string>& aq_strings,
        const std::vector<std::string>& metadata_strings,
        std::vector<char>& filter_ids_map) {
            
        }
