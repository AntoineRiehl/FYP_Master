// frontend/src/utils/atlasFiltering.ts

import type {
    AtlasNode
} from "../types/atlas";

import type {
    AtlasFilters
} from "../types/filters";


// =====================================================
// DEFAULT FILTERS
// =====================================================

export const DEFAULT_ATLAS_FILTERS: AtlasFilters = {

    domains: [],

    reviewsOnly: false,

    balanceMode: "all"

};


// =====================================================
// FILTER ATLAS
// =====================================================

export function filterAtlas(

    nodes: AtlasNode[],

    filters: AtlasFilters

): AtlasNode[] {


    return nodes.filter(

        node => {


            // -------------------------------------------------
            // DOMAIN
            // -------------------------------------------------

            if (

                filters.domains.length > 0

                &&

                !filters.domains.includes(
                    node.domain
                )

            ) {

                return false;

            }


            // -------------------------------------------------
            // REVIEWS
            // -------------------------------------------------

            if (
                filters.reviewsOnly
            ) {

                const hasReviewEmbedding = (

                    node.enrichment
                        ?.has_review_embedding

                    === true

                );


                if (
                    !hasReviewEmbedding
                ) {

                    return false;

                }

            }


            // -------------------------------------------------
            // DOMAIN BALANCE
            // -------------------------------------------------
            //
            // balanceMode is not handled here.
            //
            // This function performs inclusion/exclusion
            // filtering only. Any balancing / sampling of
            // domains can remain in the corresponding atlas
            // layout logic.
            // -------------------------------------------------


            return true;

        }

    );

}


// =====================================================
// AVAILABLE DOMAINS
// =====================================================

export function getAvailableDomains(

    nodes: AtlasNode[]

): string[] {


    return Array.from(

        new Set(

            nodes

                .map(
                    node =>
                        node.domain
                )

                .filter(
                    Boolean
                )

        )

    ).sort();

}


// =====================================================
// AVAILABLE CATEGORIES
// =====================================================

export function getAvailableCategories(

    nodes: AtlasNode[],

    domains: string[] = []

): string[] {


    const filteredNodes =

        domains.length === 0

            ?

        nodes

            :

        nodes.filter(

            node =>
                domains.includes(
                    node.domain
                )

        );


    const categories = (

        filteredNodes.flatMap(

            node =>
                node.text.categories

        )

    );


    return Array.from(

        new Set(

            categories.filter(
                Boolean
            )

        )

    ).sort();

}