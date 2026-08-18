// frontend/src/components/AtlasLayout.tsx


import {
    useEffect,
    useMemo,
    useState
} from "react";


import UniverseCanvas
    from "./UniverseCanvas";

import Sidebar
    from "./Sidebar";

import DetailsPanel
    from "./DetailsPanel";


import type {
    AtlasData,
    AtlasNode
} from "../types/atlas";


import type {
    ColorMode
} from "../config/colorMode";


import type {
    AtlasFilters
} from "../types/filters";


import {
    filterNodes,
    getAvailableDomains
} from "../utils/filterNodes";


// =====================================================
// PROPS
// =====================================================

type Props = {

    data: AtlasData;

    atlasName: string;

    onAtlasChange:
        (name: string) => void;

};


// =====================================================
// COMPONENT
// =====================================================

export default function AtlasLayout({

    data,

    atlasName,

    onAtlasChange

}: Props) {


    // =================================================
    // AVAILABLE DOMAINS
    // =================================================

    const availableDomains =
        useMemo(

            () =>
                getAvailableDomains(
                    data.atlas
                ),

            [data.atlas]

        );


    // =================================================
    // SELECTED NODE
    // =================================================

    const [
        selectedNode,
        setSelectedNode
    ] =
    useState<AtlasNode | null>(
        null
    );


    // =================================================
    // SEARCH
    // =================================================

    const [
        searchQuery,
        setSearchQuery
    ] =
    useState("");


    // =================================================
    // COLOUR MODE
    // =================================================

    const [
        colorMode,
        setColorMode
    ] =
    useState<ColorMode>(
        "cluster"
    );


    // =================================================
    // FILTERS
    // =================================================

    const [
        filters,
        setFilters
    ] =
    useState<AtlasFilters>({

        domains: [],

        reviewsOnly: false,

        balanceMode: "all"

    });


    // =================================================
    // KEEP FILTER DOMAINS IN SYNC WITH ATLAS
    // =================================================

    useEffect(() => {

        setFilters(
            current => ({

                ...current,

                domains:
                    availableDomains

            })
        );

    }, [
        atlasName,
        availableDomains
    ]);


    // =================================================
    // RESET ATLAS-SPECIFIC STATE
    // =================================================

    useEffect(() => {

        setSelectedNode(null);

        setSearchQuery("");

    }, [
        atlasName
    ]);


    // =================================================
    // FILTERED NODES
    // =================================================

    const filteredNodes =
        useMemo(

            () =>
                filterNodes(

                    data.atlas,

                    filters

                ),

            [
                data.atlas,
                filters
            ]

        );


    // =================================================
    // FILTERED ATLAS DATA
    // =================================================

    const filteredData =
        useMemo(

            () => ({

                ...data,

                atlas:
                    filteredNodes

            }),

            [
                data,
                filteredNodes
            ]

        );


    // =================================================
    // SEARCH RESULTS
    // =================================================

    const searchResults =

        searchQuery.length < 2

        ?

        []

        :

        filteredNodes

            .filter(
                node =>

                    node.title
                        .toLowerCase()
                        .includes(
                            searchQuery
                                .toLowerCase()
                        )

            )

            .slice(
                0,
                10
            );


    // =================================================
    // SELECT NODE
    // =================================================

    function handleSelectNode(
        node: AtlasNode | null
    ) {

        setSelectedNode(
            node
        );

    }


    // =================================================
    // UPDATE FILTERS
    // =================================================

    function handleFiltersChange(

        nextFilters: AtlasFilters

    ) {

        setFilters(
            nextFilters
        );


        // If the currently selected item
        // is no longer visible, clear it.

        if (
            selectedNode
            &&
            !filterNodes(
                [selectedNode],
                nextFilters
            ).length
        ) {

            setSelectedNode(
                null
            );

        }

    }


    // =================================================
    // RENDER
    // =================================================

    return (

        <div

            style={{

                display: "flex",

                width: "100vw",

                height: "100vh",

                overflow: "hidden",

                background: "#050816",

                color: "white"

            }}

        >


            {/* =================================================
                LEFT SIDEBAR
            ================================================= */}

            <Sidebar

                atlasName={
                    atlasName
                }

                onAtlasChange={
                    onAtlasChange
                }

                searchQuery={
                    searchQuery
                }

                onSearchChange={
                    setSearchQuery
                }

                searchResults={
                    searchResults
                }

                onSelectResult={
                    handleSelectNode
                }

                colorMode={
                    colorMode
                }

                onColorModeChange={
                    setColorMode
                }

                filters={
                    filters
                }

                availableDomains={
                    availableDomains
                }

                onFiltersChange={
                    handleFiltersChange
                }

            />


            {/* =================================================
                MAIN CANVAS
            ================================================= */}

            <div

                style={{

                    flex: 1,

                    position: "relative"

                }}

            >

                <UniverseCanvas

                    data={
                        filteredData
                    }

                    colorMode={
                        colorMode
                    }

                    onSelect={
                        handleSelectNode
                    }

                />

            </div>


            {/* =================================================
                DETAILS PANEL
            ================================================= */}

            <DetailsPanel

                node={
                    selectedNode
                }

            />


        </div>

    );

}