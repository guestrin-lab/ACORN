# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

add_executable(demo_imi_flat EXCLUDE_FROM_ALL demo_imi_flat.cpp)
target_link_libraries(demo_imi_flat PRIVATE faiss)

add_executable(demo_imi_pq EXCLUDE_FROM_ALL demo_imi_pq.cpp)
target_link_libraries(demo_imi_pq PRIVATE faiss)

add_executable(demo_ivfpq_indexing EXCLUDE_FROM_ALL demo_ivfpq_indexing.cpp)
target_link_libraries(demo_ivfpq_indexing PRIVATE faiss)

add_executable(demo_nndescent EXCLUDE_FROM_ALL demo_nndescent.cpp)
target_link_libraries(demo_nndescent PRIVATE faiss)

add_executable(demo_sift1M EXCLUDE_FROM_ALL demo_sift1M.cpp)
target_link_libraries(demo_sift1M PRIVATE faiss)

add_executable(demo_weighted_kmeans EXCLUDE_FROM_ALL demo_weighted_kmeans.cpp)
target_link_libraries(demo_weighted_kmeans PRIVATE faiss)

add_executable(demo_residual_quantizer EXCLUDE_FROM_ALL demo_residual_quantizer.cpp)
target_link_libraries(demo_residual_quantizer PRIVATE faiss)

# add_executable(demo_new_test EXCLUDE_FROM_ALL demo_new_test.cpp)
# target_link_libraries(demo_new_test PRIVATE faiss)

# add_executable(demo_test_search EXCLUDE_FROM_ALL demo_test_search.cpp)
# target_link_libraries(demo_test_search PRIVATE faiss)

# add_executable(demo_test_search_small EXCLUDE_FROM_ALL demo_test_search_small.cpp)
# target_link_libraries(demo_test_search_small PRIVATE faiss)

# add_executable(demo_test_hybrid_small EXCLUDE_FROM_ALL demo_test_hybrid_small.cpp)
# target_link_libraries(demo_test_hybrid_small PRIVATE faiss)

# add_executable(demo_test_hybrid_large EXCLUDE_FROM_ALL demo_test_hybrid_large.cpp)
# target_link_libraries(demo_test_hybrid_large PRIVATE faiss)

# add_executable(make_indices EXCLUDE_FROM_ALL make_indices.cpp)
# target_link_libraries(make_indices PRIVATE faiss)

# add_executable(make_sift_indices EXCLUDE_FROM_ALL make_sift_indices.cpp)
# target_link_libraries(make_sift_indices PRIVATE faiss)

# add_executable(benchmark EXCLUDE_FROM_ALL benchmark.cpp)
# target_link_libraries(benchmark PRIVATE faiss)

add_executable(utils EXCLUDE_FROM_ALL utils.cpp)
target_link_libraries(utils PRIVATE faiss)

# add_executable(correlation EXCLUDE_FROM_ALL correlation.cpp)
# target_link_libraries(correlation PRIVATE faiss)


# add_executable(check_queries EXCLUDE_FROM_ALL check_queries.cpp)
# target_link_libraries(check_queries PRIVATE faiss)

# add_executable(trace_query EXCLUDE_FROM_ALL trace_query.cpp)
# target_link_libraries(trace_query PRIVATE faiss)

# add_executable(make_debug_index EXCLUDE_FROM_ALL make_debug_index.cpp)
# target_link_libraries(make_debug_index PRIVATE faiss)

# add_executable(print_edges EXCLUDE_FROM_ALL print_edges.cpp)
# target_link_libraries(print_edges PRIVATE faiss)

# add_executable(profile_query EXCLUDE_FROM_ALL profile_query.cpp)
# target_link_libraries(profile_query PRIVATE faiss)

# add_executable(prefilter EXCLUDE_FROM_ALL prefilter.cpp)
# target_link_libraries(prefilter PRIVATE faiss)

# add_executable(make_tripclick_indices EXCLUDE_FROM_ALL make_tripclick_indices.cpp)
# target_link_libraries(make_tripclick_indices PRIVATE faiss)

# add_executable(test_tripclick_indices EXCLUDE_FROM_ALL test_tripclick_indices.cpp)
# target_link_libraries(test_tripclick_indices PRIVATE faiss)


# add_executable(trace_tripclick_query EXCLUDE_FROM_ALL trace_tripclick_query.cpp)
# target_link_libraries(trace_tripclick_query PRIVATE faiss)

# add_executable(test_tripclick_dates EXCLUDE_FROM_ALL test_tripclick_dates.cpp)
# target_link_libraries(test_tripclick_dates PRIVATE faiss)

# add_executable(make_tripclick_dates_indices EXCLUDE_FROM_ALL make_tripclick_dates_indices.cpp)
# target_link_libraries(make_tripclick_dates_indices PRIVATE faiss)


# add_executable(make_tripclick_oracle_indices EXCLUDE_FROM_ALL make_tripclick_oracle_indices.cpp)
# target_link_libraries(make_tripclick_oracle_indices PRIVATE faiss)


# add_executable(make_laion_indices EXCLUDE_FROM_ALL make_laion_indices.cpp)
# target_link_libraries(make_laion_indices PRIVATE faiss)

# add_executable(test_laion_indices EXCLUDE_FROM_ALL test_laion_indices.cpp)
# target_link_libraries(test_laion_indices PRIVATE faiss)

# add_executable(test_laion_arb_pred EXCLUDE_FROM_ALL test_laion_arb_pred.cpp)
# target_link_libraries(test_laion_arb_pred PRIVATE faiss)

add_executable(test_acorn EXCLUDE_FROM_ALL test_acorn.cpp)
target_link_libraries(test_acorn PRIVATE faiss)

add_executable(test_efsearch_params EXCLUDE_FROM_ALL test_efsearch_params.cpp)
target_link_libraries(test_efsearch_params PRIVATE faiss)
