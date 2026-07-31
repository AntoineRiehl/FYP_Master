//frontend/src/App.tsx

import { useState } from "react";

import AtlasLayout from "./components/AtlasLayout";
import { useAtlas } from "./hooks/useAtlas";


function App() {

    const [atlasName, setAtlasName] =
        useState("movies");


    const atlas =
        useAtlas(atlasName);



    if (!atlas) {

        return (

            <div
                style={{
                    width:"100vw",
                    height:"100vh",
                    display:"flex",
                    justifyContent:"center",
                    alignItems:"center",
                    background:"#050816",
                    color:"white"
                }}
            >
                Loading atlas...

            </div>

        );

    }



    return (

        <AtlasLayout
            data={atlas}
            atlasName={atlasName}
            onAtlasChange={setAtlasName}
        />

    );

}


export default App;