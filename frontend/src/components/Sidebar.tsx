// frontend/src/components/Sidebar.tsx

// frontend/src/components/Sidebar.tsx


import type {
    AtlasNode
} from "../types/atlas";


import {
    ATLAS_REGISTRY
} from "../config/atlasRegistry";


import type {
    ColorMode
} from "../config/colorMode";



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
    onColorModeChange

}: Props) {


    return (

        <aside

            style={{

                width: "260px",

                padding: "20px",

                background: "#0b1020",

                borderRight:
                    "1px solid rgba(255,255,255,0.1)",

                boxSizing: "border-box",

                overflowY: "auto"

            }}

        >


            {/* =================================================
                TITLE
            ================================================= */}

            <h2

                style={{

                    marginTop: 0,

                    fontSize: "20px"

                }}

            >

                Atlas Explorer

            </h2>



            {/* =================================================
                ATLAS SELECTOR
            ================================================= */}

            <section

                style={{

                    marginTop: "30px"

                }}

            >

                <h3

                    style={{

                        fontSize: "14px",

                        opacity: 0.7

                    }}

                >

                    Atlas

                </h3>



                <select

                    value={atlasName}

                    onChange={(e) =>

                        onAtlasChange(
                            e.target.value
                        )

                    }

                    style={{

                        width: "100%",

                        padding: "8px",

                        background: "#111827",

                        color: "white",

                        border:
                            "1px solid rgba(255,255,255,0.1)",

                        borderRadius: "4px"

                    }}

                >

                    {

                        ATLAS_REGISTRY.map(
                            (atlas) => (

                                <option

                                    key={atlas.id}

                                    value={atlas.id}

                                >

                                    {atlas.label}

                                </option>

                            )
                        )

                    }

                </select>


            </section>



            {/* =================================================
                COLOR MODE
            ================================================= */}

            <section

                style={{

                    marginTop: "30px"

                }}

            >

                <h3

                    style={{

                        fontSize: "14px",

                        opacity: 0.7

                    }}

                >

                    Color by

                </h3>



                <select

                    value={colorMode}

                    onChange={(e) => {

                        const value =
                            e.target.value as ColorMode;

                        onColorModeChange(
                            value
                        );

                    }}

                    style={{

                        width: "100%",

                        padding: "8px",

                        background: "#111827",

                        color: "white",

                        border:
                            "1px solid rgba(255,255,255,0.1)",

                        borderRadius: "4px"

                    }}

                >

                    <option value="domain">

                        Domain

                    </option>


                    <option value="cluster">

                        Cluster

                    </option>


                    <option value="category">

                        Category

                    </option>


                    <option value="feel">

                        Feel

                    </option>

                </select>


            </section>



            {/* =================================================
                SEARCH
            ================================================= */}

            <section

                style={{

                    marginTop: "30px"

                }}

            >

                <h3

                    style={{

                        fontSize: "14px",

                        opacity: 0.7

                    }}

                >

                    Search

                </h3>



                <input

                    value={searchQuery}

                    onChange={(e) =>

                        onSearchChange(
                            e.target.value
                        )

                    }

                    placeholder="Search..."

                    style={{

                        width: "100%",

                        padding: "8px",

                        background: "#111827",

                        color: "white",

                        boxSizing: "border-box",

                        border:
                            "1px solid rgba(255,255,255,0.1)",

                        borderRadius: "4px"

                    }}

                />



                {/* =================================================
                    SEARCH RESULTS
                ================================================= */}

                {

                    searchResults.length > 0 && (

                        <div

                            style={{

                                marginTop: "10px"

                            }}

                        >

                            {

                                searchResults.map(
                                    (node) => (

                                        <div

                                            key={node.id}

                                            onClick={() =>

                                                onSelectResult(
                                                    node
                                                )

                                            }

                                            style={{

                                                padding: "6px",

                                                cursor: "pointer",

                                                borderBottom:
                                                    "1px solid rgba(255,255,255,0.1)"

                                            }}

                                        >

                                            {node.title}

                                        </div>

                                    )
                                )

                            }

                        </div>

                    )

                }


            </section>


        </aside>

    );

}