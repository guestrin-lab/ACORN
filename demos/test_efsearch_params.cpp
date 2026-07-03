// Regression demo for the params->efSearch bug in ACORN::hybrid_search /
// ACORN::search.
//
// Loads the ACORN-gamma index persisted by test_acorn from ./tmp, then runs
// the same filtered query batch with SearchParametersACORN::efSearch set to
// 16, 64 and 256.
//
// BEFORE the fix: all three rows are identical (the candidate heap was sized
// by the member efSearch frozen at construction, so the search-time parameter
// had no effect above the default of 16).
// AFTER the fix: recall and distance computations scale with efSearch.
//
// Usage (after running `test_acorn 1000000 12 sift1M 32 64` once to persist
// the index and with testing_data/ populated as described in the README):
//   ./build/demos/test_efsearch_params 1000000 12 sift1M 32 64

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <sys/stat.h>
#include <sys/time.h>
#include <unistd.h>

#include <faiss/IndexACORN.h>
#include <faiss/index_io.h>

#include <faiss/Index.h>
#include <faiss/impl/platform_macros.h>
#include <assert.h>
#include <math.h>
#include <fstream>
#include <iostream>
#include <numeric>
#include <set>
#include <sstream>
#include <thread>
#include <nlohmann/json.hpp>
#include "utils.cpp"

int main(int argc, char* argv[]) {
    if (argc < 6) {
        fprintf(stderr, "Syntax: %s <N> <gamma> <dataset> <M> <M_beta>\n", argv[0]);
        exit(1);
    }
    size_t N = strtoul(argv[1], nullptr, 10);
    int gamma = atoi(argv[2]);
    std::string dataset = argv[3];
    // argv[4] = M, argv[5] = M_beta: kept for CLI symmetry with test_acorn

    int k = 10;
    int n_centroids = gamma;
    std::string assignment_type = "rand";
    int alpha = 0;

    // ---- data (same conventions as test_acorn) ----
    std::vector<int> metadata = load_ab(dataset, n_centroids, assignment_type, N);
    metadata.resize(N);

    size_t d = 0, nq = 0;
    float* xq = fvecs_read(get_file_name(dataset, /*is_base=*/false).c_str(), &d, &nq);
    std::vector<int> aq = load_aq(dataset, n_centroids, alpha, N);

    int gt_size = 100;
    std::vector<faiss::idx_t> gt = load_gt(dataset, gamma, alpha, assignment_type, N);

    // ---- load the persisted ACORN-gamma index ----
    std::stringstream ss;
    ss << "./tmp/hybrid_" << (int)(N / 1000 / 1000) << "m_nc=" << n_centroids
       << "_assignment=" << assignment_type << "_alpha=" << alpha << ".json";
    faiss::Index* raw = faiss::read_index(ss.str().c_str());
    auto* index = dynamic_cast<faiss::IndexACORNFlat*>(raw);
    FAISS_THROW_IF_NOT_MSG(index, "loaded index is not IndexACORNFlat");
    index->acorn.metadata = metadata.data(); // not serialized; re-bind
    printf("loaded %s (ntotal=%ld, gamma=%d, member efSearch=%d)\n",
           ss.str().c_str(), index->ntotal, index->acorn.gamma,
           index->acorn.efSearch);

    // ---- filter bitmap ----
    std::vector<char> filter_ids_map(nq * N);
    for (size_t q = 0; q < nq; q++) {
        char* row = filter_ids_map.data() + q * N;
        for (size_t b = 0; b < N; b++) {
            row[b] = (char)(metadata[b] == aq[q]);
        }
    }

    // ---- the sweep: identical rows == the bug ----
    printf("\n%-10s %-12s %-22s\n", "efSearch", "Recall@10", "dist comps / query");
    for (int efs : {16, 64, 256}) {
        faiss::SearchParametersACORN params;
        params.efSearch = efs;

        std::vector<faiss::idx_t> nns(k * nq, -1);
        std::vector<float> dis(k * nq);
        faiss::acorn_stats.reset();
        index->search(nq, xq, k, dis.data(), nns.data(),
                      filter_ids_map.data(), &params);

        float recall = compute_recall(gt, gt_size, nns, nq, k) / k;
        double dpq = faiss::acorn_stats.n1
                ? (double)faiss::acorn_stats.n3 / faiss::acorn_stats.n1
                : 0.0;
        printf("%-10d %-12.4f %-22.2f\n", efs, recall, dpq);
    }

    delete index;
    delete[] xq;
    return 0;
}
