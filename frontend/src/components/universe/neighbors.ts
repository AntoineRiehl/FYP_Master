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


// =====================================================
// POPULARITY BANDS
// =====================================================

export type PopularityBand =

    | "mainstream"
    | "middle"
    | "niche"
    | "unknown";


export type DomainPopularityThreshold = {

    domain: string;

    population: number;

    /*
     * Bottom 20% threshold.
     */

    nicheMax: number;

    /*
     * Top 20% threshold.
     */

    mainstreamMin: number;

};


export type DomainPopularityThresholds =

    Record<
        string,
        DomainPopularityThreshold
    >;


// =====================================================
// NEIGHBOUR ANALYSIS
// =====================================================

export type NeighborAnalysis = {

    /*
     * The N closest items overall, regardless of domain.
     *
     * These are the nodes connected visually
     * on the canvas.
     */

    nearest: NeighborResult[];


    /*
     * Closest entity from the same domain.
     */

    closestSameDomain:
        NeighborResult | null;


    /*
     * Closest entity for every OTHER domain.
     *
     * Example:
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


    /*
     * Closest item classified as mainstream.
     *
     * Mainstream means top 20% popularity
     * WITHIN THAT ITEM'S OWN DOMAIN.
     */

    closestMainstream:
        NeighborResult | null;


    /*
     * Closest item classified as niche.
     *
     * Niche means bottom 20% popularity
     * WITHIN THAT ITEM'S OWN DOMAIN.
     */

    closestNiche:
        NeighborResult | null;

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
     * if object references change later.
     */

    return (

        a === b

        ||

        a.id === b.id

    );

}



// =====================================================
// POPULARITY VALUE
// =====================================================

function getPopularity(

    node: AtlasNode

): number | null {


    const popularity =

        node.statistics.popularity;


    if (

        popularity === null

        ||

        popularity === undefined

        ||

        !Number.isFinite(
            popularity
        )

    ) {

        return null;

    }


    return popularity;

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
     * 250,000 nodes makes the scan cheaper.
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
     * This array is tiny — normally only five or six
     * items — so sorting it repeatedly is inexpensive.
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
// PERCENTILE VALUE
// =====================================================

function percentileValue(

    sortedValues: number[],

    percentile: number

): number {


    if (
        sortedValues.length === 0
    ) {

        return 0;

    }


    if (
        sortedValues.length === 1
    ) {

        return sortedValues[0];

    }


    const position =

        (
            sortedValues.length
            -
            1
        )

        *

        percentile;


    const lowerIndex =

        Math.floor(
            position
        );


    const upperIndex =

        Math.ceil(
            position
        );


    if (
        lowerIndex === upperIndex
    ) {

        return sortedValues[
            lowerIndex
        ];

    }


    const weight =

        position
        -
        lowerIndex;


    return (

        sortedValues[
            lowerIndex
        ]

        *

        (
            1
            -
            weight
        )

        +

        sortedValues[
            upperIndex
        ]

        *

        weight

    );

}



// =====================================================
// BUILD DOMAIN POPULARITY THRESHOLDS
// =====================================================

export function buildDomainPopularityThresholds(

    nodes: AtlasNode[],

    nichePercentile = 0.20,

    mainstreamPercentile = 0.80

): DomainPopularityThresholds {


    // =================================================
    // COLLECT POPULARITY BY DOMAIN
    // =================================================

    const valuesByDomain =

        new Map<
            string,
            number[]
        >();


    for (
        const node
        of nodes
    ) {


        const popularity =

            getPopularity(
                node
            );


        if (
            popularity === null
        ) {

            continue;

        }


        let values =

            valuesByDomain.get(
                node.domain
            );


        if (!values) {


            values = [];


            valuesByDomain.set(

                node.domain,

                values

            );

        }


        values.push(
            popularity
        );

    }



    // =================================================
    // CALCULATE THRESHOLDS
    // =================================================

    const thresholds:

        DomainPopularityThresholds = {};


    for (
        const [
            domain,
            values
        ]
        of valuesByDomain
    ) {


        /*
         * A tiny population does not give us a useful
         * mainstream / niche distinction.
         */

        if (
            values.length < 5
        ) {

            continue;

        }


        values.sort(

            (a, b) =>
                a - b

        );


        const minimum =
            values[0];


        const maximum =
            values[
                values.length - 1
            ];


        /*
         * If every item has exactly the same popularity,
         * there is no meaningful popularity band.
         */

        if (
            minimum === maximum
        ) {

            continue;

        }


        thresholds[
            domain
        ] = {

            domain,

            population:
                values.length,

            nicheMax:

                percentileValue(

                    values,

                    nichePercentile

                ),

            mainstreamMin:

                percentileValue(

                    values,

                    mainstreamPercentile

                )

        };

    }


    return thresholds;

}



// =====================================================
// GET POPULARITY BAND
// =====================================================

export function getPopularityBand(

    node: AtlasNode,

    thresholds:
        DomainPopularityThresholds

): PopularityBand {


    const popularity =

        getPopularity(
            node
        );


    if (
        popularity === null
    ) {

        return "unknown";

    }


    const domainThreshold =

        thresholds[
            node.domain
        ];


    if (
        !domainThreshold
    ) {

        return "unknown";

    }


    if (

        popularity
        >=
        domainThreshold.mainstreamMin

    ) {

        return "mainstream";

    }


    if (

        popularity
        <=
        domainThreshold.nicheMax

    ) {

        return "niche";

    }


    return "middle";

}



// =====================================================
// ANALYSE NEIGHBOURS
// =====================================================

export function analyzeNeighbors(

    nodes: AtlasNode[],

    selectedNode: AtlasNode,

    nearestCount = 5,

    popularityThresholds?:
        DomainPopularityThresholds

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
    // MAINSTREAM ANALOGUE
    // =================================================

    let closestMainstreamWorking: {

        node: AtlasNode;

        distanceSquared: number;

    } | null = null;



    // =================================================
    // NICHE ANALOGUE
    // =================================================

    let closestNicheWorking: {

        node: AtlasNode;

        distanceSquared: number;

    } | null = null;



    // =================================================
    // SINGLE PASS THROUGH ATLAS
    // =================================================

    for (
        const node
        of nodes
    ) {


        // ---------------------------------------------
        // IGNORE SELECTED ITEM ITSELF
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
        // POPULARITY ANALOGUES
        // =================================================

        if (
            popularityThresholds
        ) {


            const band =

                getPopularityBand(

                    node,

                    popularityThresholds

                );


            // -----------------------------------------
            // MAINSTREAM
            // -----------------------------------------

            if (
                band === "mainstream"
            ) {


                if (

                    closestMainstreamWorking === null

                    ||

                    distanceSquared
                    <
                    closestMainstreamWorking
                        .distanceSquared

                ) {


                    closestMainstreamWorking = {

                        node,

                        distanceSquared

                    };

                }

            }


            // -----------------------------------------
            // NICHE
            // -----------------------------------------

            if (
                band === "niche"
            ) {


                if (

                    closestNicheWorking === null

                    ||

                    distanceSquared
                    <
                    closestNicheWorking
                        .distanceSquared

                ) {


                    closestNicheWorking = {

                        node,

                        distanceSquared

                    };

                }

            }

        }



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


            /*
             * We already handled popularity bands above.
             *
             * Nothing below this point is needed for
             * same-domain nodes.
             */

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
    // CONVERT TOP-N DISTANCES
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



    // =================================================
    // CONVERT SAME-DOMAIN DISTANCE
    // =================================================

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



    // =================================================
    // CONVERT CROSS-DOMAIN DISTANCES
    // =================================================

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
    // CONVERT MAINSTREAM ANALOGUE
    // =================================================

    const closestMainstream:

        NeighborResult | null =

        closestMainstreamWorking

        ?

        {

            node:
                closestMainstreamWorking.node,

            distance:

                Math.sqrt(

                    closestMainstreamWorking
                        .distanceSquared

                )

        }

        :

        null;



    // =================================================
    // CONVERT NICHE ANALOGUE
    // =================================================

    const closestNiche:

        NeighborResult | null =

        closestNicheWorking

        ?

        {

            node:
                closestNicheWorking.node,

            distance:

                Math.sqrt(

                    closestNicheWorking
                        .distanceSquared

                )

        }

        :

        null;



    // =================================================
    // RESULT
    // =================================================

    return {

        nearest,

        closestSameDomain,

        closestByDomain,

        closestMainstream,

        closestNiche

    };

}