if (TARGET implicit_shader::implicit_shader)
    return()
endif()

include(FetchContent)
FetchContent_Declare(
        implicit_shader
        GIT_REPOSITORY git@github.com:qnzhou/implicit_shader.git
        GIT_TAG main
)

FetchContent_MakeAvailable(implicit_shader)
