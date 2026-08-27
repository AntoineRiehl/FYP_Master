// frontend/src/components/universe/neighbors.ts

import type {
    AtlasNode
} from "../../types/atlas";



// =====================================================
// TYPES
// =====================================================

export type NeighborResult = {

    node: AtlasNode;

    /*
     * Euclidean distance in the displayed atlas'
     * 2D world coordinate system.
     *
     * This is NOT a source-space semantic distance.
     */

    distance: number;

};


export type NeighborAnalysis = {

    /*
     * The N closest items overall, regardless of domain.
     *
     * These are the nodes we will connect visually
     * on the canvas.
     */

    nearest: NeighborResult[];


    /*
     * Closest entity from the same domain.
     *
     * Useful later in the sidebar.
     */

    closestSameDomain:
        NeighborResult | null;


    /*
     * Closest entity for every OTHER domain.
     *
     * Example for selected Movie:
     *
     * {
     *     music: {...},
     *     restaurants: {...}
     * }
     */

    closestByDomain:
        Record<
            string,
            NeighborResult
        >;

};



// =====================================================
// SAME NODE
// =====================================================

function isSameNode(

    a: AtlasNode,

    b: AtlasNode

): boolean {


    /*
     * Usually selectedNode is the exact object from
     * data.atlas, but ID comparison makes this robust
     * if React/state creates another reference later.
     */

    return (

        a === b

        ||

        a.id === b.id

    );

}



// =====================================================
// DISTANCE
// =====================================================

function atlasDistanceSquared(

    a: AtlasNode,

    b: AtlasNode

): number {


    const dx =

        a.position.x
        -
        b.position.x;


    const dy =

        a.position.y
        -
        b.position.y;


    /*
     * Squared distance is sufficient while searching.
     *
     * Avoiding Math.sqrt() for every one of potentially
     * 250,000 nodes makes the scan slightly cheaper.
     */

    return (

        dx * dx

        +

        dy * dy

    );

}



// =====================================================
// INSERT INTO TOP-K
// =====================================================

function insertNearest(

    nearest: {

        node: AtlasNode;

        distanceSquared: number;

    }[],

    node: AtlasNode,

    distanceSquared: number,

    maxNeighbors: number

) {


    nearest.push({

        node,

        distanceSquared

    });


    /*
     * This array is tiny — normally only 5 or 6 items —
     * so sorting it is cheap.
     */

    nearest.sort(

        (a, b) =>

            a.distanceSquared
            -
            b.distanceSquared

    );


    if (

        nearest.length
        >
        maxNeighbors

    ) {

        nearest.pop();

    }

}



// =====================================================
// ANALYSE NEIGHBOURS
// =====================================================

export function analyzeNeighbors(

    nodes: AtlasNode[],

    selectedNode: AtlasNode,

    nearestCount = 5

): NeighborAnalysis {


    // =================================================
    // TOP N OVERALL
    // =================================================

    const nearestWorking: {

        node: AtlasNode;

        distanceSquared: number;

    }[] = [];



    // =================================================
    // SAME DOMAIN
    // =================================================

    let closestSameDomainWorking: {

        node: AtlasNode;

        distanceSquared: number;

    } | null = null;



    // =================================================
    // OTHER DOMAINS
    // =================================================

    const closestByDomainWorking:

        Record<
            string,
            {
                node: AtlasNode;
                distanceSquared: number;
            }
        >

        = {};



    // =================================================
    // SINGLE PASS THROUGH ATLAS
    // =================================================

    for (const node of nodes) {


        // ---------------------------------------------
        // Ignore selected item itself
        // ---------------------------------------------

        if (

            isSameNode(

                node,

                selectedNode

            )

        ) {

            continue;

        }



        // ---------------------------------------------
        // DISTANCE
        // ---------------------------------------------

        const distanceSquared =

            atlasDistanceSquared(

                selectedNode,

                node

            );



        // =================================================
        // NEAREST OVERALL
        // =================================================

        insertNearest(

            nearestWorking,

            node,

            distanceSquared,

            nearestCount

        );



        // =================================================
        // CLOSEST SAME DOMAIN
        // =================================================

        if (

            node.domain
            ===
            selectedNode.domain

        ) {


            if (

                closestSameDomainWorking === null

                ||

                distanceSquared
                <
                closestSameDomainWorking
                    .distanceSquared

            ) {


                closestSameDomainWorking = {

                    node,

                    distanceSquared

                };

            }


            continue;

        }



        // =================================================
        // CLOSEST OTHER DOMAIN
        // =================================================

        const current =

            closestByDomainWorking[
                node.domain
            ];


        if (

            !current

            ||

            distanceSquared
            <
            current.distanceSquared

        ) {


            closestByDomainWorking[
                node.domain
            ] = {

                node,

                distanceSquared

            };

        }

    }



    // =================================================
    // CONVERT DISTANCES
    // =================================================

    const nearest:
        NeighborResult[] =

        nearestWorking.map(

            result => ({

                node:
                    result.node,

                distance:
                    Math.sqrt(
                        result.distanceSquared
                    )

            })

        );



    const closestSameDomain:

        NeighborResult | null =

        closestSameDomainWorking

        ?

        {

            node:
                closestSameDomainWorking.node,

            distance:
                Math.sqrt(

                    closestSameDomainWorking
                        .distanceSquared

                )

        }

        :

        null;



    const closestByDomain:

        Record<
            string,
            NeighborResult
        >

        = {};



    for (

        const [
            domain,
            result
        ]

        of Object.entries(
            closestByDomainWorking
        )

    ) {


        closestByDomain[
            domain
        ] = {

            node:
                result.node,

            distance:
                Math.sqrt(
                    result.distanceSquared
                )

        };

    }



    // =================================================
    // RESULT
    // =================================================

    return {

        nearest,

        closestSameDomain,

        closestByDomain

    };

}