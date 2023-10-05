function(CheckHasModule Module)
    find_package(${Module} QUIET)
    if(NOT DEFINED ${Module}_DIR)
        set(HAS_MODULE_${Module} TRUE PARENT_SCOPE)
    elseif(${Module}_DIR)
        set(HAS_MODULE_${Module} TRUE PARENT_SCOPE)
    else()
        set(HAS_MODULE_${Module} FALSE PARENT_SCOPE)
    endif()
endfunction()

CheckHasModule(benchmark)


add_subdirectory(tensor_times_vector)
add_subdirectory(vb_example)
add_subdirectory(lz77)

if(${HAS_MODULE_benchmark})
    add_subdirectory(lz77_bench)
    add_subdirectory(png_reader)
endif()