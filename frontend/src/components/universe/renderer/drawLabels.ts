// frontend/src/components/universe/renderer/drawLabels.ts

import type {
    AtlasNode
} from "../../../types/atlas";


import type {
    Camera
} from "../camera";


import {
    worldToScreen
} from "../camera";


// =====================================================
// TYPES
// =====================================================

type LabelRectangle = {

    left: number;

    right: number;

    top: number;

    bottom: number;

};


// =====================================================
// LABEL BUDGET
// =====================================================

function getLabelBudget(

    visibleNodeCount: number

): number {


    /*
     * The number of labels depends primarily on how many
     * nodes are currently visible.
     *
     * This is deliberately conservative.
     *
     * At large / medium scale we want the map structure
     * to remain readable rather than attempting to label
     * every visible item.
     */


    if (visibleNodeCount > 10000) {

        return 0;

    }


    if (visibleNodeCount > 5000) {

        return 6;

    }


    if (visibleNodeCount > 2500) {

        return 10;

    }


    if (visibleNodeCount > 1200) {

        return 18;

    }


    if (visibleNodeCount > 600) {

        return 28;

    }


    if (visibleNodeCount > 300) {

        return 40;

    }


    if (visibleNodeCount > 150) {

        return 55;

    }


    if (visibleNodeCount > 75) {

        return 70;

    }


    /*
     * Once the user is very close, we can attempt to
     * display nearly every visible item.
     *
     * Collision detection still prevents unreadable
     * overlaps.
     */

    return Math.min(
        visibleNodeCount,
        100
    );

}


// =====================================================
// RECTANGLE COLLISION
// =====================================================

function rectanglesOverlap(

    a: LabelRectangle,

    b: LabelRectangle

): boolean {


    return !(

        a.right < b.left

        ||

        a.left > b.right

        ||

        a.bottom < b.top

        ||

        a.top > b.bottom

    );

}


// =====================================================
// LABEL COLLISION
// =====================================================

function collidesWithExistingLabel(

    rectangle: LabelRectangle,

    occupied: LabelRectangle[]

): boolean {


    for (const existing of occupied) {


        if (

            rectanglesOverlap(
                rectangle,
                existing
            )

        ) {

            return true;

        }

    }


    return false;

}


// =====================================================
// DRAW NODE LABELS
// =====================================================

export function drawNodeLabels(

    ctx: CanvasRenderingContext2D,

    canvas: HTMLCanvasElement,

    camera: Camera,

    nodes: AtlasNode[],

    totalVisibleNodeCount: number

) {


    // =================================================
    // BUDGET
    // =================================================

    const labelBudget =

        getLabelBudget(
            totalVisibleNodeCount
        );


    if (labelBudget <= 0) {

        return;

    }



    // =================================================
    // PRIORITISE IMPORTANT NODES
    // =================================================

    /*
     * Only the already-rendered/capped nodes are considered
     * here.
     *
     * Larger visual.size means greater visual importance /
     * popularity in the existing atlas representation.
     */

    const candidates =

        [...nodes]

            .sort(

                (a, b) =>

                    b.visual.size
                    -
                    a.visual.size

            );



    // =================================================
    // TEXT STYLE
    // =================================================

    ctx.save();


    ctx.font =
        "11px Arial";


    ctx.textBaseline =
        "middle";


    ctx.fillStyle =
        "white";


    /*
     * A subtle dark outline makes labels readable on top
     * of coloured nodes without needing opaque boxes.
     */

    ctx.strokeStyle =
        "rgba(5, 8, 22, 0.9)";


    ctx.lineWidth =
        3;



    // =================================================
    // COLLISION STATE
    // =================================================

    const occupied:
        LabelRectangle[] = [];


    let labelsDrawn = 0;



    // =================================================
    // DRAW
    // =================================================

    for (const node of candidates) {


        if (

            labelsDrawn
            >=
            labelBudget

        ) {

            break;

        }



        const pos =

            worldToScreen(

                camera,

                canvas,

                node.position.x,

                node.position.y

            );



        // ---------------------------------------------
        // NODE RADIUS
        // ---------------------------------------------

        const radius =

            Math.max(

                1.5,

                Math.sqrt(
                    node.visual.size
                )

            );



        // ---------------------------------------------
        // LABEL POSITION
        // ---------------------------------------------

        const labelX =

            pos.x
            +
            radius
            +
            5;


        const labelY =

            pos.y;



        // ---------------------------------------------
        // TEXT SIZE
        // ---------------------------------------------

        const metrics =

            ctx.measureText(
                node.title
            );


        const width =

            metrics.width;


        const height = 13;



        // ---------------------------------------------
        // RECTANGLE
        // ---------------------------------------------

        const rectangle:
            LabelRectangle = {


                left:
                    labelX - 2,

                right:
                    labelX + width + 2,

                top:
                    labelY
                    -
                    height / 2
                    -
                    2,

                bottom:
                    labelY
                    +
                    height / 2
                    +
                    2

            };



        // ---------------------------------------------
        // SCREEN BOUNDS
        // ---------------------------------------------

        /*
         * Do not spend label budget on labels that would
         * be mostly outside the canvas.
         */

        if (

            rectangle.right < 0

            ||

            rectangle.left > canvas.width

            ||

            rectangle.bottom < 0

            ||

            rectangle.top > canvas.height

        ) {

            continue;

        }



        // ---------------------------------------------
        // COLLISION
        // ---------------------------------------------

        if (

            collidesWithExistingLabel(

                rectangle,

                occupied

            )

        ) {

            continue;

        }



        // ---------------------------------------------
        // DRAW LABEL
        // ---------------------------------------------

        /*
         * strokeText acts as a dark halo around the text.
         */

        ctx.strokeText(

            node.title,

            labelX,

            labelY

        );


        ctx.fillText(

            node.title,

            labelX,

            labelY

        );



        occupied.push(
            rectangle
        );


        labelsDrawn++;

    }



    ctx.restore();

}