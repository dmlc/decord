# SPDX-License-Identifier: Apache‑2.0
include_guard(GLOBAL)

# ---------------------------------------------------------------------------
# decord_option(<var> "help" <default | generator‑expression> [IF <cond>])
# ---------------------------------------------------------------------------
function(decord_option VAR DESCRIPTION DEFAULT_VALUE)
    cmake_parse_arguments(OPT "" "IF" "" ${ARGN})

    # Evaluate the gating condition (defaults to true)
    if(NOT DEFINED OPT_IF)
        set(_cond TRUE)
    else()
        set(_cond ${OPT_IF})
    endif()

    if(_cond)
        # The user may have set the cache variable already
        if(NOT DEFINED ${VAR})
            option(${VAR} "${DESCRIPTION}" ${DEFAULT_VALUE})
        endif()
    else()
        # Remove it from the cache so it does not show up in GUIs
        unset(${VAR} CACHE)
    endif()
endfunction()

# ---------------------------------------------------------------------------
# assign_source_group(<group> file1 [file2 …])
# ---------------------------------------------------------------------------
function(assign_source_group GROUP)
    foreach(src IN LISTS ARGN)
        if(IS_ABSOLUTE "${src}")
            file(RELATIVE_PATH rel "${CMAKE_CURRENT_SOURCE_DIR}" "${src}")
        else()
            set(rel "${src}")
        endif()
        get_filename_component(path "${rel}" PATH)
        string(REPLACE "/" "\\" path_msvc "${path}")
        source_group("${GROUP}\\${path_msvc}" FILES "${src}")
    endforeach()
endfunction()