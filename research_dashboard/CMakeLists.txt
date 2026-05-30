cmake_minimum_required(VERSION 3.14)
project(elara_research_dashboard LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(elara_research_snapshot
    cpp/main.cpp
)

target_include_directories(elara_research_snapshot PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/third_party
)

if(MSVC)
    target_compile_options(elara_research_snapshot PRIVATE /W4)
else()
    target_compile_options(elara_research_snapshot PRIVATE -Wall -Wextra -Wpedantic)
endif()
