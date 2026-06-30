add_library(optimizer OBJECT)

target_sources(
    optimizer PRIVATE
    parallelism_optimizer.cpp
)
