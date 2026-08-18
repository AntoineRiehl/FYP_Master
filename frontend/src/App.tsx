//frontend/src/App.tsx

import { useState } from "react";

import AtlasLayout from "./components/AtlasLayout";

import { useAtlas } from "./hooks/useAtlas";


// =====================================================
// APP
// =====================================================

function App() {


    // -------------------------------------------------
    // SELECTED ATLAS
    // -------------------------------------------------

    const [
        atlasName,
        setAtlasName
    ] =
        useState("movies");


    // -------------------------------------------------
    // LOAD ATLAS
    // -------------------------------------------------

    const {
        data: atlas,
        loading,
        error
    } =
        useAtlas(
            atlasName
        );


    // =================================================
    // LOADING
    // =================================================

    if (loading) {

        return (

            <div
                style={{
                    width: "100vw",
                    height: "100vh",

                    display: "flex",

                    justifyContent: "center",

                    alignItems: "center",

                    background: "#050816",

                    color: "white",

                    fontSize: "16px"
                }}
            >

                Loading{" "}
                <strong
                    style={{
                        marginLeft: "5px"
                    }}
                >
                    {atlasName}
                </strong>
                {" "}atlas...

            </div>

        );

    }


    // =================================================
    // ERROR
    // =================================================

    if (error) {

        return (

            <div
                style={{
                    width: "100vw",
                    height: "100vh",

                    display: "flex",

                    flexDirection: "column",

                    justifyContent: "center",

                    alignItems: "center",

                    background: "#050816",

                    color: "white",

                    padding: "30px",

                    boxSizing: "border-box",

                    textAlign: "center"
                }}
            >

                <h2>
                    Failed to load atlas
                </h2>


                <p
                    style={{
                        opacity: 0.7,
                        maxWidth: "700px"
                    }}
                >
                    Could not load the{" "}
                    <strong>
                        {atlasName}
                    </strong>
                    {" "}atlas.
                </p>


                <p
                    style={{
                        opacity: 0.5,
                        fontSize: "13px",
                        maxWidth: "800px"
                    }}
                >
                    {error.message}
                </p>


                <button
                    onClick={() =>
                        window.location.reload()
                    }

                    style={{
                        marginTop: "15px",

                        padding: "8px 16px",

                        background: "#111827",

                        color: "white",

                        border:
                            "1px solid rgba(255,255,255,0.2)",

                        borderRadius: "4px",

                        cursor: "pointer"
                    }}
                >

                    Retry

                </button>

            </div>

        );

    }


    // =================================================
    // NO DATA
    // =================================================

    if (!atlas) {

        return (

            <div
                style={{
                    width: "100vw",
                    height: "100vh",

                    display: "flex",

                    justifyContent: "center",

                    alignItems: "center",

                    background: "#050816",

                    color: "white"
                }}
            >

                No atlas data available.

            </div>

        );

    }


    // =================================================
    // MAIN APPLICATION
    // =================================================

    return (

        <AtlasLayout

            data={atlas}

            atlasName={atlasName}

            onAtlasChange={
                setAtlasName
            }

        />

    );

}


export default App;