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
} from "../types/atlas";


import type {
    AtlasFilters
} from "../types/filters";


import {
    filterNodes,
    getAvailableDomains
} from "../utils/filterNodes";


import {
    analyzeNeighbors
} from "./universe/neighbors";


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
    // NODE TO FOCUS
    // =================================================

    const [
        focusNode,
        setFocusNode
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

        setFocusNode(null);

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
    // NEIGHBOUR ANALYSIS
    // =================================================

    /*
     * This runs only when:
     *
     * - selection changes
     * - filtered atlas population changes
     *
     * It does NOT run once per animation frame.
     */

    const neighborAnalysis =
        useMemo(

            () => {

                if (!selectedNode) {

                    return null;

                }


                return analyzeNeighbors(

                    filteredNodes,

                    selectedNode,

                    5

                );

            },

            [
                filteredNodes,
                selectedNode
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
    // SELECT SEARCH RESULT
    // =================================================

    function handleSelectSearchResult(
        node: AtlasNode
    ) {

        setSelectedNode(
            node
        );


        setFocusNode(
            node
        );


        setSearchQuery("");

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


        // If selected item becomes invisible,
        // clear the selection.

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


        // If focused item becomes invisible,
        // clear the focus request.

        if (
            focusNode
            &&
            !filterNodes(
                [focusNode],
                nextFilters
            ).length
        ) {

            setFocusNode(
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
                    handleSelectSearchResult
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

                    selectedNode={
                        selectedNode
                    }

                    neighbors={
                        neighborAnalysis?.nearest
                        ??
                        []
                    }

                    focusNode={
                        focusNode
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