// frontend/src/components/AtlasLayout.tsx

import {
    useState
} from "react";


import UniverseCanvas from "./UniverseCanvas";
import Sidebar from "./Sidebar";
import DetailsPanel from "./DetailsPanel";


import type {
    AtlasData,
    AtlasNode
} from "../types/atlas";



type Props = {

    data: AtlasData;

    atlasName:string;

    onAtlasChange:
        (name:string)=>void;

};



export default function AtlasLayout({

    data,

    atlasName,

    onAtlasChange

}:Props){


    const [
        selectedNode,
        setSelectedNode
    ] =
    useState<AtlasNode|null>(null);



    const [
        searchQuery,
        setSearchQuery
    ] =
    useState("");



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

            .slice(0,10);



    return (

        <div

            style={{

                display:"flex",

                width:"100vw",

                height:"100vh",

                overflow:"hidden",

                background:"#050816",

                color:"white"

            }}

        >


            <Sidebar

                atlasName={atlasName}

                onAtlasChange={onAtlasChange}

                searchQuery={searchQuery}

                onSearchChange={setSearchQuery}

                searchResults={searchResults}

                onSelectResult={setSelectedNode}

            />



            <div

                style={{

                    flex:1,

                    position:"relative"

                }}

            >


                <UniverseCanvas

                    data={data}

                    onSelect={setSelectedNode}

                />


            </div>



            <DetailsPanel

                node={selectedNode}

            />


        </div>

    );

}