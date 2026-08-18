// frontend/src/components/Sidebar.tsx

import type {
    AtlasNode
} from "../types/atlas";

import type {
    ColorMode
} from "../config/colorMode";

import type {
    AtlasFilters
} from "../types/filters";


import {
    AtlasSelector,
    ColorModeSelector,
    FilterPanel,
    SearchPanel
} from "./sidebar_components";


// =====================================================
// PROPS
// =====================================================

type Props = {

    atlasName: string;

    onAtlasChange:
        (name: string) => void;


    searchQuery: string;

    onSearchChange:
        (query: string) => void;

    searchResults: AtlasNode[];

    onSelectResult:
        (node: AtlasNode) => void;


    colorMode: ColorMode;

    onColorModeChange:
        (mode: ColorMode) => void;


    filters: AtlasFilters;

    availableDomains: string[];

    onFiltersChange:
        (filters: AtlasFilters) => void;

};


// =====================================================
// COMPONENT
// =====================================================

export default function Sidebar({

    atlasName,
    onAtlasChange,

    searchQuery,
    onSearchChange,

    searchResults,
    onSelectResult,

    colorMode,
    onColorModeChange,

    filters,
    availableDomains,
    onFiltersChange

}: Props) {

    return (

        <aside

            style={{

                width: "280px",

                padding: "20px",

                background: "#0b1020",

                borderRight:
                    "1px solid rgba(255,255,255,0.1)",

                boxSizing: "border-box",

                overflowY: "auto"

            }}

        >

            <h2

                style={{

                    marginTop: 0,

                    fontSize: "20px"

                }}

            >
                Atlas Explorer
            </h2>


            <AtlasSelector

                atlasName={atlasName}

                onAtlasChange={onAtlasChange}

            />


            <ColorModeSelector

                colorMode={colorMode}

                onColorModeChange={
                    onColorModeChange
                }

            />


            <FilterPanel

                filters={filters}

                availableDomains={
                    availableDomains
                }

                onFiltersChange={
                    onFiltersChange
                }

            />


            <SearchPanel

                searchQuery={searchQuery}

                onSearchChange={
                    onSearchChange
                }

                searchResults={
                    searchResults
                }

                onSelectResult={
                    onSelectResult
                }

            />

        </aside>

    );

}