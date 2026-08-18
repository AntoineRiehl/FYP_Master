// frontend/src/utils/filterNodes.ts


import type {
    AtlasNode
} from "../types/atlas";

import type {
    AtlasFilters
} from "../types/filters";


// =====================================================
// POPULARITY SCORE
// =====================================================

function getPopularityScore(
    node: AtlasNode
): number {

    const popularity =
        node.statistics.popularity;


    if (
        popularity !== null
        &&
        popularity !== undefined
        &&
        Number.isFinite(popularity)
    ) {

        return popularity;

    }


    const ratingCount =
        node.statistics.rating_count;


    if (
        ratingCount !== null
        &&
        ratingCount !== undefined
        &&
        Number.isFinite(ratingCount)
    ) {

        return ratingCount;

    }


    const rating =
        node.statistics.rating;


    if (
        rating !== null
        &&
        rating !== undefined
        &&
        Number.isFinite(rating)
    ) {

        return rating;

    }


    return 0;

}


// =====================================================
// DOMAIN FILTER
// =====================================================

function filterByDomain(

    nodes: AtlasNode[],

    selectedDomains: string[]

): AtlasNode[] {


    // Empty selection means:
    // show all domains.

    if (
        selectedDomains.length === 0
    ) {

        return nodes;

    }


    const allowedDomains =
        new Set(
            selectedDomains
        );


    return nodes.filter(
        node =>
            allowedDomains.has(
                node.domain
            )
    );

}


// =====================================================
// REVIEW FILTER
// =====================================================

function filterByReviews(

    nodes: AtlasNode[],

    reviewsOnly: boolean

): AtlasNode[] {


    if (!reviewsOnly) {

        return nodes;

    }


    return nodes.filter(
        node =>
            Array.isArray(
                node.text.reviews
            )
            &&
            node.text.reviews.length > 0
    );

}


// =====================================================
// BALANCED DOMAIN FILTER
// =====================================================

function balanceDomains(

    nodes: AtlasNode[]

): AtlasNode[] {


    // Group nodes by their original domain.

    const grouped =
        new Map<
            string,
            AtlasNode[]
        >();


    for (const node of nodes) {

        const existing =
            grouped.get(
                node.domain
            );


        if (existing) {

            existing.push(node);

        }
        else {

            grouped.set(
                node.domain,
                [node]
            );

        }

    }


    // If there is only one domain,
    // balancing has no effect.

    if (grouped.size <= 1) {

        return nodes;

    }


    const domainArrays =
        Array.from(
            grouped.values()
        );


    // The smallest domain determines
    // the target size.

    const targetSize =
        Math.min(
            ...domainArrays.map(
                items =>
                    items.length
            )
        );


    const balanced: AtlasNode[] = [];


    for (
        const domainNodes
        of domainArrays
    ) {


        // Sort by popularity so that the
        // most representative / popular
        // items are retained.

        const sorted =
            [...domainNodes].sort(
                (
                    a,
                    b
                ) =>
                    getPopularityScore(b)
                    -
                    getPopularityScore(a)
            );


        balanced.push(
            ...sorted.slice(
                0,
                targetSize
            )
        );

    }


    return balanced;

}


// =====================================================
// MAIN FILTER FUNCTION
// =====================================================

export function filterNodes(

    nodes: AtlasNode[],

    filters: AtlasFilters

): AtlasNode[] {


    // -------------------------------------------------
    // DOMAIN
    // -------------------------------------------------

    let filtered =
        filterByDomain(

            nodes,

            filters.domains

        );


    // -------------------------------------------------
    // REVIEWS
    // -------------------------------------------------

    filtered =
        filterByReviews(

            filtered,

            filters.reviewsOnly

        );


    // -------------------------------------------------
    // BALANCE
    // -------------------------------------------------

    if (
        filters.balanceMode ===
        "balanced"
    ) {

        filtered =
            balanceDomains(
                filtered
            );

    }


    return filtered;

}


// =====================================================
// AVAILABLE DOMAINS
// =====================================================

export function getAvailableDomains(

    nodes: AtlasNode[]

): string[] {


    return Array.from(

        new Set(

            nodes.map(
                node =>
                    node.domain
            )

        )

    ).sort();

}