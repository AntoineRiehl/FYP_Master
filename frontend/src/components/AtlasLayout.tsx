// frontend/src/components/AtlasLayout.tsx

import { useState } from "react";

import UniverseCanvas from "./UniverseCanvas";
import Sidebar from "./Sidebar";
import DetailsPanel from "./DetailsPanel";

import type {
    AtlasData,
    AtlasNode
} from "../types/atlas";

type Props = {

    data: AtlasData;

    atlasName: string;

    onAtlasChange:
        (name:string)=>void;

};

export default function AtlasLayout({

    data,
    atlasName,
    onAtlasChange

}:Props){

    const [
        hoveredNode,
        setHoveredNode
    ] =
        useState<AtlasNode | null>(null);

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
            />

            <div
                style={{
                    flex:1,
                    position:"relative"
                }}
            >

                <UniverseCanvas
                    data={data}
                    onHover={setHoveredNode}
                />

            </div>

            <DetailsPanel
                node={hoveredNode}
            />

        </div>

    );

}