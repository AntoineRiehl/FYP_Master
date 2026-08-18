// frontend/src/components/AtlasLayout.tsx

// frontend/src/components/AtlasLayout.tsx

import {
    useEffect,
    useState
} from "react";


import UniverseCanvas from "./UniverseCanvas";
import Sidebar from "./Sidebar";
import DetailsPanel from "./DetailsPanel";


import type {
    AtlasData,
    AtlasNode
} from "../types/atlas";


import type {
    ColorMode
} from "../config/colorMode";


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
    // SELECTED NODE
    // =================================================

    const [
        selectedNode,
        setSelectedNode
    ] =
    useState<AtlasNode | null>(null);


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
    useState<ColorMode>("cluster");


    // =================================================
    // RESET SELECTION WHEN ATLAS CHANGES
    // =================================================

    useEffect(() => {

        setSelectedNode(null);

        setSearchQuery("");

    }, [atlasName]);


    // =================================================
    // SEARCH RESULTS
    // =================================================

    const searchResults =

        searchQuery.length < 2

        ?

        []

        :

        data.atlas

            .filter(node =>

                node.title

                    .toLowerCase()

                    .includes(
                        searchQuery.toLowerCase()
                    )

            )

            .slice(0, 10);


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

                atlasName={atlasName}

                onAtlasChange={onAtlasChange}

                searchQuery={searchQuery}

                onSearchChange={setSearchQuery}

                searchResults={searchResults}

                onSelectResult={setSelectedNode}

                colorMode={colorMode}

                onColorModeChange={setColorMode}

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

                    data={data}

                    colorMode={colorMode}

                    onSelect={setSelectedNode}

                />

            </div>


            {/* =================================================
                DETAILS PANEL
            ================================================= */}

            <DetailsPanel

                node={selectedNode}

            />


        </div>

    );

}