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
    AtlasNode,
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
    analyzeNeighbors,
    buildDomainPopularityThresholds
} from "./universe/neighbors";


import {
    buildClusterStatsIndex,
    getNodeClusterStats
} from "../utils/clusterStats";


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
    // =====================================================

    const availableDomains =
        useMemo(

            () =>
                getAvailableDomains(
                    data.atlas
                ),

            [
                data.atlas
            ]

        );


    // =================================================
    // SELECTED NODE
    // =====================================================

    const [
        selectedNode,
        setSelectedNode
    ] =
        useState<AtlasNode | null>(
            null
        );


    // =================================================
    // NODE TO FOCUS
    // =====================================================

    const [
        focusNode,
        setFocusNode
    ] =
        useState<AtlasNode | null>(
            null
        );


    // =================================================
    // SEARCH
    // =====================================================

    const [
        searchQuery,
        setSearchQuery
    ] =
        useState("");


    // =================================================
    // COLOUR MODE
    // =====================================================

    const [
        colorMode,
        setColorMode
    ] =
        useState<ColorMode>(
            "cluster"
        );


    // =================================================
    // FILTERS
    // =====================================================

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
    // KEEP FILTER DOMAINS IN SYNC
    // =====================================================

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
    // =====================================================

    useEffect(() => {

        setSelectedNode(
            null
        );

        setFocusNode(
            null
        );

        setSearchQuery(
            ""
        );

    }, [
        atlasName
    ]);


    // =================================================
    // FILTERED NODES
    // =====================================================

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
    // =====================================================

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
    // POPULARITY THRESHOLDS
    // =====================================================

    /*
     * Mainstream / niche definitions are calculated
     * from the COMPLETE atlas.
     *
     * They therefore do not change when the user
     * temporarily changes filters.
     */

    const popularityThresholds =
        useMemo(

            () =>
                buildDomainPopularityThresholds(

                    data.atlas,

                    0.20,

                    0.80

                ),

            [
                data.atlas
            ]

        );


    // =================================================
    // CLUSTER STATS INDEX
    // =====================================================

    /*
     * Cluster statistics also describe the COMPLETE
     * atlas, not the temporarily filtered view.
     *
     * This is calculated once when an atlas loads.
     */

    const clusterStatsIndex =
        useMemo(

            () =>
                buildClusterStatsIndex(
                    data.atlas
                ),

            [
                data.atlas
            ]

        );


    // =================================================
    // SELECTED NODE CLUSTER STATS
    // =====================================================

    const selectedClusterStats =
        useMemo(

            () => {

                if (!selectedNode) {

                    return null;

                }


                return getNodeClusterStats(

                    selectedNode,

                    clusterStatsIndex

                );

            },

            [
                selectedNode,
                clusterStatsIndex
            ]

        );


    // =================================================
    // NEIGHBOUR ANALYSIS
    // =====================================================

    /*
     * Neighbour candidates use filteredNodes so:
     *
     * - canvas links correspond to visible entities;
     * - sidebar exploration respects active filters.
     *
     * Popularity thresholds themselves remain based on
     * the complete atlas.
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

                    5,

                    popularityThresholds

                );

            },

            [
                filteredNodes,
                selectedNode,
                popularityThresholds
            ]

        );


    // =================================================
    // SEARCH RESULTS
    // =====================================================

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
    // SELECT NODE FROM CANVAS
    // =====================================================

    function handleSelectNode(

        node: AtlasNode | null

    ) {

        /*
         * Canvas selection should not automatically
         * move the camera.
         */

        setSelectedNode(
            node
        );


        /*
         * Clear an old focus request so a future
         * sidebar navigation can focus the same item
         * again if necessary.
         */

        setFocusNode(
            null
        );

    }


    // =================================================
    // SELECT SEARCH RESULT
    // =====================================================

    function handleSelectSearchResult(

        node: AtlasNode

    ) {

        setSelectedNode(
            node
        );


        setFocusNode(
            node
        );


        setSearchQuery(
            ""
        );

    }


    // =================================================
    // SELECT RELATED ITEM
    // =====================================================

    /*
     * Used by clickable neighbours and Discovery links
     * in the DetailsPanel.
     *
     * Unlike a canvas click, sidebar navigation should
     * move the camera to the newly selected item.
     */

    function handleSelectRelatedNode(

        node: AtlasNode

    ) {

        setSelectedNode(
            node
        );


        setFocusNode(
            node
        );


        setSearchQuery(
            ""
        );

    }


    // =================================================
    // UPDATE FILTERS
    // =====================================================

    function handleFiltersChange(

        nextFilters: AtlasFilters

    ) {

        setFilters(
            nextFilters
        );


        // ---------------------------------------------
        // CLEAR INVISIBLE SELECTION
        // ---------------------------------------------

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


        // ---------------------------------------------
        // CLEAR INVISIBLE FOCUS
        // ---------------------------------------------

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
    // =====================================================

    return (

        <div
            style={{

                display:
                    "flex",

                width:
                    "100vw",

                height:
                    "100vh",

                overflow:
                    "hidden",

                background:
                    "#050816",

                color:
                    "white"

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

                    flex:
                        1,

                    position:
                        "relative"

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

                atlasName={
                    atlasName
                }

                neighborAnalysis={
                    neighborAnalysis
                }

                clusterStats={
                    selectedClusterStats
                }

                onSelectNode={
                    handleSelectRelatedNode
                }

            />


        </div>

    );

}