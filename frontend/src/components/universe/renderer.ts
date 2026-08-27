// frontend/src/components/universe/renderer.ts

import type {
    AtlasData,
    AtlasNode
} from "../../types/atlas";


import type {
    Camera
} from "./camera";


import type {
    NeighborResult
} from "./neighbors";


import {
    getLOD
} from "./lod";


import {
    drawRegions
} from "./renderer/drawRegions";


import {
    drawNodes
} from "./renderer/drawNodes";


import {
    drawNeighborLinks
} from "./renderer/drawNeighborLinks";


import {
    drawSelection
} from "./renderer/drawSelection";


import {
    detectHover
} from "./renderer/hover";


import type {
    ColorMode
} from "../../config/colorMode";


// =====================================================
// RENDER UNIVERSE
// =====================================================

export function renderUniverse(

    ctx: CanvasRenderingContext2D,

    canvas: HTMLCanvasElement,

    camera: Camera,

    data: AtlasData,

    mouse: {
        x: number;
        y: number;
    },

    colorMode: ColorMode,

    selectedNode: AtlasNode | null,

    neighbors: NeighborResult[]

): AtlasNode | null {


    // =================================================
    // CLEAR
    // =================================================

    ctx.clearRect(

        0,

        0,

        canvas.width,

        canvas.height

    );


    // =================================================
    // LEVEL OF DETAIL
    // =================================================

    const mode =

        getLOD(
            camera.zoom
        );


    // =================================================
    // REGIONS
    // =================================================

    if (mode === 0) {


        drawRegions(

            ctx,

            canvas,

            camera,

            data.regions

        );

    }


    // =================================================
    // FULL / LANDMARK ATLAS
    // =================================================

    else {


        // =============================================
        // NEIGHBOUR LINKS
        // =============================================

        /*
         * Links are drawn first so nodes remain visually
         * above them.
         */

        if (

            selectedNode

            &&

            neighbors.length > 0

        ) {


            drawNeighborLinks(

                ctx,

                canvas,

                camera,

                selectedNode,

                neighbors

            );

        }


        // =============================================
        // NORMAL NODES
        // =============================================

        drawNodes(

            ctx,

            canvas,

            camera,

            data.atlas,

            mode,

            colorMode,

            selectedNode

        );

    }


    // =================================================
    // SELECTED NODE
    // =================================================

    /*
     * Selection is drawn last so it always sits above
     * the normal atlas.
     */

    if (selectedNode) {


        drawSelection(

            ctx,

            canvas,

            camera,

            selectedNode,

            mode,

            colorMode

        );

    }


    // =================================================
    // HOVER
    // =================================================

    const hovered =

        detectHover(

            canvas,

            camera,

            data.atlas,

            mouse

        );


    // =================================================
    // TOOLTIP
    // =================================================

    if (hovered) {


        ctx.fillStyle =
            "rgba(0,0,0,0.75)";


        ctx.fillRect(

            mouse.x + 10,

            mouse.y + 10,

            230,

            28

        );


        ctx.fillStyle =
            "white";


        ctx.font =
            "12px Arial";


        ctx.fillText(

            hovered.title,

            mouse.x + 15,

            mouse.y + 28

        );

    }


    return hovered;

}