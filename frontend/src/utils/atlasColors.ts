//frontend/src/utils/atlasColors.ts

import type {
    AtlasNode,
    ColorMode
} from "../types/atlas";


// =====================================================
// COLOR PALETTE
// =====================================================

const COLOR_PALETTE = [

    "#ff6b6b",
    "#4dabf7",
    "#51cf66",
    "#fcc419",
    "#845ef7",
    "#ff922b",
    "#20c997",
    "#e64980",
    "#15aabf",
    "#94d82d",
    "#be4bdb",
    "#f06595",
    "#228be6",
    "#40c057",
    "#fab005",
    "#7950f2"

];


// =====================================================
// HASH
// =====================================================

function hashString(
    value: string
): number {

    let hash = 0;


    for (
        let i = 0;
        i < value.length;
        i++
    ) {

        hash =
            (
                hash << 5
            )
            -
            hash
            +
            value.charCodeAt(i);

        hash |= 0;

    }


    return Math.abs(hash);

}


// =====================================================
// COLOR KEY
// =====================================================

function getColorKey(

    node: AtlasNode,

    mode: ColorMode

): string {


    switch(mode){

        case "domain":

            return (
                node.domain ||
                "unknown"
            );


        case "cluster":

            return (
                node.visual.cluster !== null &&
                node.visual.cluster !== undefined

                    ?

                `cluster:${node.visual.cluster}`

                    :

                "cluster:unknown"
            );


        case "category": {

            const category =
                node.text.categories[0];


            if(!category){

                return (
                    `${node.domain}:unknown`
                );

            }


            return (
                `${node.domain}:${category}`
            );

        }

    }

}


// =====================================================
// COLOR
// =====================================================

export function getNodeColor(

    node: AtlasNode,

    mode: ColorMode

): string {


    const key =
        getColorKey(
            node,
            mode
        );


    const index =
        hashString(key)
        %
        COLOR_PALETTE.length;


    return COLOR_PALETTE[index];

}