// frontend/src/utils/clusterStats.ts

import type {
    AtlasNode
} from "../types/atlas";


// =====================================================
// TYPES
// =====================================================

export type ClusterDomainComposition = {

    domain: string;

    count: number;

    /*
     * Share from 0 → 1.
     */

    share: number;

};


export type ClusterCentre = {

    x: number;

    y: number;

};


export type ClusterSummary = {

    clusterId: number;

    clusterLabel: string | null;

    size: number;

    centre: ClusterCentre;

    domainComposition:
        ClusterDomainComposition[];

};


export type CompetingCluster = {

    clusterId: number;

    clusterLabel: string | null;

    size: number;

    /*
     * Distance from the selected item to the
     * competing cluster centre in the displayed
     * 2D atlas.
     */

    distanceToCentre: number;

};


export type NodeClusterStats = {

    // -------------------------------------------------
    // CLUSTER IDENTITY
    // -------------------------------------------------

    clusterId: number;

    clusterLabel: string | null;

    clusterSize: number;


    // -------------------------------------------------
    // DOMAIN COMPOSITION
    // -------------------------------------------------

    domainComposition:
        ClusterDomainComposition[];


    // -------------------------------------------------
    // POPULARITY
    // -------------------------------------------------

    /*
     * Popularity rank is calculated ONLY against
     * members of the SAME DOMAIN inside the cluster.
     *
     * This avoids invalid comparisons such as:
     *
     * MovieLens rating counts
     * vs
     * Last.fm popularity
     * vs
     * Yelp popularity.
     *
     * Example:
     *
     * popularityTopPercent = 14
     *
     * means:
     *
     * "Top 14% most popular Movies in this region"
     *
     * if the selected node is a Movie.
     */

    popularityTopPercent:
        number | null;

    popularityRank:
        number | null;

    /*
     * Number of same-domain cluster members
     * with a usable popularity value.
     */

    popularityPopulation:
        number;


    // -------------------------------------------------
    // CENTRALITY
    // -------------------------------------------------

    /*
     * Distance from the selected item to the centre
     * of its assigned cluster in the displayed
     * 2D atlas.
     */

    distanceToClusterCentre:
        number;


    /*
     * Example:
     *
     * centralityTopPercent = 8
     *
     * means:
     *
     * "Top 8% most central items in this region."
     *
     * Smaller distance to the cluster centre means
     * greater centrality.
     */

    centralityTopPercent:
        number;

    centralityRank:
        number;


    // -------------------------------------------------
    // NEAREST COMPETING REGION
    // -------------------------------------------------

    nearestCompetingCluster:
        CompetingCluster | null;


    /*
     * Relative centroid proximity.
     *
     * Example:
     *
     * currentClusterProximity = 0.68
     * competingClusterProximity = 0.32
     *
     * may be displayed as:
     *
     * Current region     68%
     * Nearest region     32%
     *
     * IMPORTANT:
     *
     * These are NOT membership probabilities.
     * They are derived purely from distances to
     * cluster centroids in the displayed 2D atlas.
     */

    currentClusterProximity:
        number | null;

    competingClusterProximity:
        number | null;


    // -------------------------------------------------
    // BOUNDARY DESCRIPTION
    // -------------------------------------------------

    boundaryStatus:
        string;

};


// =====================================================
// INTERNAL TYPES
// =====================================================

type ClusterAccumulator = {

    clusterId: number;

    members: AtlasNode[];

    sumX: number;

    sumY: number;

    domainCounts:
        Map<string, number>;

    labelCounts:
        Map<string, number>;

};


type InternalClusterSummary = {

    clusterId: number;

    clusterLabel: string | null;

    size: number;

    centre: ClusterCentre;

    domainComposition:
        ClusterDomainComposition[];


    /*
     * Popularity distributions are stored separately
     * for each domain.
     *
     * Example:
     *
     * movies -> [...]
     * music -> [...]
     * restaurants -> [...]
     *
     * Each array is sorted ascending.
     */

    popularityValuesByDomain:
        Map<
            string,
            number[]
        >;


    /*
     * Distances of all cluster members from the
     * cluster centre.
     *
     * Sorted ascending:
     *
     * smaller distance = more central.
     */

    centralityDistances:
        number[];

};


export type ClusterStatsIndex = {

    clusters:
        Map<
            number,
            InternalClusterSummary
        >;

};


// =====================================================
// VALID CLUSTER
// =====================================================

function getClusterId(

    node: AtlasNode

): number | null {


    const cluster =
        node.visual.cluster;


    if (
        cluster === null
        ||
        cluster === undefined
        ||
        !Number.isFinite(cluster)
        ||
        cluster < 0
    ) {

        return null;

    }


    return cluster;

}


// =====================================================
// POPULARITY
// =====================================================

function getPopularity(

    node: AtlasNode

): number | null {


    const value =
        node.statistics.popularity;


    if (
        value === null
        ||
        value === undefined
        ||
        !Number.isFinite(value)
    ) {

        return null;

    }


    return value;

}


// =====================================================
// DISTANCE
// =====================================================

function distance(

    x1: number,

    y1: number,

    x2: number,

    y2: number

): number {


    return Math.hypot(

        x1 - x2,

        y1 - y2

    );

}


// =====================================================
// MOST COMMON LABEL
// =====================================================

function getMostCommonLabel(

    counts:
        Map<string, number>

): string | null {


    let bestLabel:
        string | null = null;


    let bestCount =
        0;


    for (
        const [
            label,
            count
        ]
        of counts
    ) {


        if (
            count > bestCount
        ) {

            bestLabel =
                label;

            bestCount =
                count;

        }

    }


    return bestLabel;

}


// =====================================================
// BINARY SEARCH
// =====================================================

/*
 * Returns the first index where:
 *
 * value >= target
 */

function lowerBound(

    values: number[],

    target: number

): number {


    let low =
        0;


    let high =
        values.length;


    while (
        low < high
    ) {


        const middle =

            Math.floor(

                (
                    low
                    +
                    high
                )
                /
                2

            );


        if (
            values[middle]
            <
            target
        ) {

            low =
                middle + 1;

        }

        else {

            high =
                middle;

        }

    }


    return low;

}


/*
 * Returns the first index where:
 *
 * value > target
 */

function upperBound(

    values: number[],

    target: number

): number {


    let low =
        0;


    let high =
        values.length;


    while (
        low < high
    ) {


        const middle =

            Math.floor(

                (
                    low
                    +
                    high
                )
                /
                2

            );


        if (
            values[middle]
            <=
            target
        ) {

            low =
                middle + 1;

        }

        else {

            high =
                middle;

        }

    }


    return low;

}


// =====================================================
// POPULARITY RANK
// =====================================================

function rankPopularity(

    sortedValues: number[],

    value: number

): {

    rank: number;

    topPercent: number;

} {


    /*
     * Larger popularity value = better ranking.
     *
     * Count how many items have a STRICTLY greater
     * popularity value.
     */

    const firstGreaterIndex =

        upperBound(

            sortedValues,

            value

        );


    const morePopularCount =

        sortedValues.length

        -

        firstGreaterIndex;


    const rank =

        morePopularCount
        +
        1;


    const topPercent =

        (
            rank
            /
            sortedValues.length
        )
        *
        100;


    return {

        rank,

        topPercent:
            Math.min(
                100,
                topPercent
            )

    };

}


// =====================================================
// CENTRALITY RANK
// =====================================================

function rankCentrality(

    sortedDistances: number[],

    selectedDistance: number

): {

    rank: number;

    topPercent: number;

} {


    /*
     * Smaller distance = greater centrality.
     *
     * Count how many cluster members are STRICTLY
     * closer to the cluster centre.
     */

    const moreCentralCount =

        lowerBound(

            sortedDistances,

            selectedDistance

        );


    const rank =

        moreCentralCount
        +
        1;


    const topPercent =

        (
            rank
            /
            sortedDistances.length
        )
        *
        100;


    return {

        rank,

        topPercent:
            Math.min(
                100,
                topPercent
            )

    };

}


// =====================================================
// BOUNDARY STATUS
// =====================================================

function getBoundaryStatus(

    currentClusterProximity:
        number | null

): string {


    if (
        currentClusterProximity === null
    ) {

        return "No competing region";

    }


    /*
     * This is relative centroid proximity,
     * NOT membership probability.
     */


    if (
        currentClusterProximity < 0.5
    ) {

        return (
            "Closer to neighbouring region centre"
        );

    }


    if (
        currentClusterProximity < 0.55
    ) {

        return (
            "Very near region boundary"
        );

    }


    if (
        currentClusterProximity < 0.65
    ) {

        return (
            "Near region boundary"
        );

    }


    if (
        currentClusterProximity < 0.8
    ) {

        return (
            "Inside region"
        );

    }


    return (
        "Far from competing region"
    );

}


// =====================================================
// BUILD CLUSTER STATS INDEX
// =====================================================

export function buildClusterStatsIndex(

    nodes: AtlasNode[]

): ClusterStatsIndex {


    // =================================================
    // FIRST PASS:
    // ACCUMULATE CLUSTER MEMBERS
    // =====================================================

    const accumulators =

        new Map<
            number,
            ClusterAccumulator
        >();


    for (
        const node
        of nodes
    ) {


        const clusterId =

            getClusterId(
                node
            );


        if (
            clusterId === null
        ) {

            continue;

        }


        let cluster =

            accumulators.get(
                clusterId
            );


        if (!cluster) {


            cluster = {

                clusterId,

                members:
                    [],

                sumX:
                    0,

                sumY:
                    0,

                domainCounts:
                    new Map(),

                labelCounts:
                    new Map()

            };


            accumulators.set(

                clusterId,

                cluster

            );

        }


        // ---------------------------------------------
        // MEMBER
        // ---------------------------------------------

        cluster.members.push(
            node
        );


        // ---------------------------------------------
        // POSITION
        // ---------------------------------------------

        cluster.sumX +=
            node.position.x;


        cluster.sumY +=
            node.position.y;


        // ---------------------------------------------
        // DOMAIN COUNT
        // ---------------------------------------------

        cluster.domainCounts.set(

            node.domain,

            (
                cluster.domainCounts.get(
                    node.domain
                )
                ??
                0
            )
            +
            1

        );


        // ---------------------------------------------
        // CLUSTER LABEL
        // ---------------------------------------------

        const label =

            node.visual.cluster_label;


        if (
            label
            &&
            label.trim()
        ) {


            cluster.labelCounts.set(

                label,

                (
                    cluster.labelCounts.get(
                        label
                    )
                    ??
                    0
                )
                +
                1

            );

        }

    }


    // =================================================
    // SECOND PASS:
    // BUILD CLUSTER SUMMARIES
    // =====================================================

    const clusters =

        new Map<
            number,
            InternalClusterSummary
        >();


    for (
        const [
            clusterId,
            accumulator
        ]
        of accumulators
    ) {


        const size =

            accumulator
                .members
                .length;


        if (
            size === 0
        ) {

            continue;

        }


        // ---------------------------------------------
        // CLUSTER CENTRE
        // ---------------------------------------------

        const centre:
            ClusterCentre = {

            x:
                accumulator.sumX
                /
                size,

            y:
                accumulator.sumY
                /
                size

        };


        // ---------------------------------------------
        // DOMAIN COMPOSITION
        // ---------------------------------------------

        const domainComposition:

            ClusterDomainComposition[] =

            Array.from(

                accumulator
                    .domainCounts
                    .entries()

            )

                .map(

                    ([
                        domain,
                        count
                    ]) => ({

                        domain,

                        count,

                        share:
                            count
                            /
                            size

                    })

                )

                .sort(

                    (a, b) =>

                        b.count
                        -
                        a.count

                );


        // ---------------------------------------------
        // POPULARITY VALUES BY DOMAIN
        // ---------------------------------------------

        const popularityValuesByDomain =

            new Map<
                string,
                number[]
            >();


        // ---------------------------------------------
        // CENTRALITY DISTANCES
        // ---------------------------------------------

        const centralityDistances:

            number[] = [];


        // ---------------------------------------------
        // PROCESS MEMBERS
        // ---------------------------------------------

        for (
            const node
            of accumulator.members
        ) {


            // =========================================
            // POPULARITY
            // =========================================

            const popularity =

                getPopularity(
                    node
                );


            if (
                popularity !== null
            ) {


                let values =

                    popularityValuesByDomain.get(
                        node.domain
                    );


                if (!values) {


                    values =
                        [];


                    popularityValuesByDomain.set(

                        node.domain,

                        values

                    );

                }


                values.push(
                    popularity
                );

            }


            // =========================================
            // CENTRALITY
            // =========================================

            centralityDistances.push(

                distance(

                    node.position.x,

                    node.position.y,

                    centre.x,

                    centre.y

                )

            );

        }


        // ---------------------------------------------
        // SORT DOMAIN POPULARITY VALUES
        // ---------------------------------------------

        for (
            const values
            of popularityValuesByDomain.values()
        ) {


            values.sort(

                (a, b) =>
                    a - b

            );

        }


        // ---------------------------------------------
        // SORT CENTRALITY DISTANCES
        // ---------------------------------------------

        centralityDistances.sort(

            (a, b) =>
                a - b

        );


        // ---------------------------------------------
        // FINAL INTERNAL SUMMARY
        // ---------------------------------------------

        clusters.set(

            clusterId,

            {

                clusterId,

                clusterLabel:
                    getMostCommonLabel(
                        accumulator.labelCounts
                    ),

                size,

                centre,

                domainComposition,

                popularityValuesByDomain,

                centralityDistances

            }

        );

    }


    return {

        clusters

    };

}


// =====================================================
// PUBLIC CLUSTER SUMMARIES
// =====================================================

export function getClusterSummaries(

    index: ClusterStatsIndex

): ClusterSummary[] {


    return Array.from(

        index.clusters.values()

    )

        .map(

            cluster => ({

                clusterId:
                    cluster.clusterId,

                clusterLabel:
                    cluster.clusterLabel,

                size:
                    cluster.size,

                centre:
                    cluster.centre,

                domainComposition:
                    cluster.domainComposition

            })

        )

        .sort(

            (a, b) =>

                a.clusterId
                -
                b.clusterId

        );

}


// =====================================================
// NODE CLUSTER STATS
// =====================================================

export function getNodeClusterStats(

    node: AtlasNode,

    index: ClusterStatsIndex

): NodeClusterStats | null {


    // =================================================
    // CURRENT CLUSTER
    // =====================================================

    const clusterId =

        getClusterId(
            node
        );


    if (
        clusterId === null
    ) {

        return null;

    }


    const cluster =

        index.clusters.get(
            clusterId
        );


    if (!cluster) {

        return null;

    }


    // =================================================
    // DISTANCE TO OWN CLUSTER CENTRE
    // =====================================================

    const distanceToClusterCentre =

        distance(

            node.position.x,

            node.position.y,

            cluster.centre.x,

            cluster.centre.y

        );


    // =================================================
    // CENTRALITY
    // =====================================================

    const centrality =

        rankCentrality(

            cluster.centralityDistances,

            distanceToClusterCentre

        );


    // =================================================
    // DOMAIN-SPECIFIC POPULARITY DISTRIBUTION
    // =====================================================

    /*
     * Crucially, only compare the selected node with
     * members of the SAME DOMAIN inside this cluster.
     */

    const domainPopularityValues =

        cluster
            .popularityValuesByDomain
            .get(
                node.domain
            )

        ??

        [];


    // =================================================
    // POPULARITY
    // =====================================================

    const popularity =

        getPopularity(
            node
        );


    let popularityTopPercent:
        number | null = null;


    let popularityRank:
        number | null = null;


    if (
        popularity !== null

        &&

        domainPopularityValues.length > 0
    ) {


        const result =

            rankPopularity(

                domainPopularityValues,

                popularity

            );


        popularityTopPercent =
            result.topPercent;


        popularityRank =
            result.rank;

    }


    // =================================================
    // NEAREST COMPETING CLUSTER
    // =====================================================

    let nearestCompeting:

        InternalClusterSummary | null = null;


    let nearestCompetingDistance =

        Infinity;


    for (
        const candidate
        of index.clusters.values()
    ) {


        if (
            candidate.clusterId
            ===
            clusterId
        ) {

            continue;

        }


        const candidateDistance =

            distance(

                node.position.x,

                node.position.y,

                candidate.centre.x,

                candidate.centre.y

            );


        if (
            candidateDistance
            <
            nearestCompetingDistance
        ) {


            nearestCompeting =
                candidate;


            nearestCompetingDistance =
                candidateDistance;

        }

    }


    // =================================================
    // RELATIVE CENTROID PROXIMITY
    // =====================================================

    let currentClusterProximity:
        number | null = null;


    let competingClusterProximity:
        number | null = null;


    let nearestCompetingCluster:
        CompetingCluster | null = null;


    if (
        nearestCompeting
        &&
        Number.isFinite(
            nearestCompetingDistance
        )
    ) {


        const totalDistance =

            distanceToClusterCentre

            +

            nearestCompetingDistance;


        if (
            totalDistance > 0
        ) {


            /*
             * If:
             *
             * own-centre distance is small
             * and
             * competing-centre distance is large
             *
             * then currentClusterProximity approaches 1.
             */

            currentClusterProximity =

                nearestCompetingDistance

                /

                totalDistance;


            competingClusterProximity =

                distanceToClusterCentre

                /

                totalDistance;

        }

        else {


            /*
             * Extremely unlikely case where the selected
             * item lies exactly on both centroids.
             */

            currentClusterProximity =
                0.5;


            competingClusterProximity =
                0.5;

        }


        nearestCompetingCluster = {

            clusterId:
                nearestCompeting.clusterId,

            clusterLabel:
                nearestCompeting.clusterLabel,

            size:
                nearestCompeting.size,

            distanceToCentre:
                nearestCompetingDistance

        };

    }


    // =================================================
    // RESULT
    // =====================================================

    return {

        // ---------------------------------------------
        // IDENTITY
        // ---------------------------------------------

        clusterId,

        clusterLabel:
            cluster.clusterLabel,

        clusterSize:
            cluster.size,


        // ---------------------------------------------
        // COMPOSITION
        // ---------------------------------------------

        domainComposition:
            cluster.domainComposition,


        // ---------------------------------------------
        // POPULARITY
        // ---------------------------------------------

        popularityTopPercent,

        popularityRank,

        popularityPopulation:
            domainPopularityValues.length,


        // ---------------------------------------------
        // CENTRALITY
        // ---------------------------------------------

        distanceToClusterCentre,

        centralityTopPercent:
            centrality.topPercent,

        centralityRank:
            centrality.rank,


        // ---------------------------------------------
        // COMPETING REGION
        // ---------------------------------------------

        nearestCompetingCluster,

        currentClusterProximity,

        competingClusterProximity,


        // ---------------------------------------------
        // BOUNDARY DESCRIPTION
        // ---------------------------------------------

        boundaryStatus:

            getBoundaryStatus(
                currentClusterProximity
            )

    };

}