cmake_minimum_required(VERSION 3.20)
project(BIMMY VERSION 1.1 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(FetchContent)

FetchContent_Declare(
  glfw
  GIT_REPOSITORY https://github.com/glfw/glfw.git
  GIT_TAG 3.4
)
FetchContent_MakeAvailable(glfw)

FetchContent_Declare(
  glm
  GIT_REPOSITORY https://github.com/g-truc/glm.git
  GIT_TAG 1.0.1
)
FetchContent_MakeAvailable(glm)

FetchContent_Declare(
  glad
  GIT_REPOSITORY https://github.com/Dav1dde/glad.git
  GIT_TAG v2.0.8
)
FetchContent_MakeAvailable(glad)
add_subdirectory(${glad_SOURCE_DIR}/cmake ${glad_BINARY_DIR}/cmake)
glad_add_library(glad REPRODUCIBLE API gl:core=3.3)

FetchContent_Declare(
  imgui
  GIT_REPOSITORY https://github.com/ocornut/imgui.git
  GIT_TAG v1.91.9b
)
FetchContent_MakeAvailable(imgui)

find_package(OpenGL REQUIRED)

add_library(imgui_lib STATIC
  ${imgui_SOURCE_DIR}/imgui.cpp
  ${imgui_SOURCE_DIR}/imgui_draw.cpp
  ${imgui_SOURCE_DIR}/imgui_widgets.cpp
  ${imgui_SOURCE_DIR}/imgui_tables.cpp
  ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
  ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp
)
target_include_directories(imgui_lib PUBLIC
  ${imgui_SOURCE_DIR}
  ${imgui_SOURCE_DIR}/backends
)
target_link_libraries(imgui_lib PUBLIC glfw)

add_executable(bimmy
  src/main.cpp
  src/renderer.cpp
  src/mesh.cpp
)

target_include_directories(bimmy PRIVATE include)
target_compile_definitions(bimmy PRIVATE
  BIMMY_SHADER_DIR="${CMAKE_CURRENT_SOURCE_DIR}/shaders"
  BIMMY_TEXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/textures"
)
target_link_libraries(bimmy PRIVATE
  OpenGL::GL
  glfw
  glad
  glm::glm
  imgui_lib
)

add_custom_command(TARGET bimmy POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:bimmy>/shaders
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          ${CMAKE_CURRENT_SOURCE_DIR}/shaders/bimmy.vert
          $<TARGET_FILE_DIR:bimmy>/shaders/bimmy.vert
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          ${CMAKE_CURRENT_SOURCE_DIR}/shaders/bimmy.frag
          $<TARGET_FILE_DIR:bimmy>/shaders/bimmy.frag
)

if(MSVC)
  target_compile_options(bimmy PRIVATE /W4)
else()
  target_compile_options(bimmy PRIVATE -Wall -Wextra -Wpedantic)
endif()
