add_subdirectory(autograd)
add_subdirectory(module)
add_subdirectory(optimizer)
add_subdirectory(mnist)

if (TINYTORCH_USE_NCCL)
    add_subdirectory(nccl)
    add_subdirectory(ddp)
endif ()
