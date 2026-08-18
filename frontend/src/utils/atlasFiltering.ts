//frontend/src/utils/atlasFiltering.ts

import type {
    AtlasFilters,
    AtlasNode
} from "../types/atlas";


// =====================================================
// DEFAULT FILTERS
// =====================================================

export const DEFAULT_ATLAS_FILTERS: AtlasFilters = {

    domains: [],

    categories: []

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

            if(
                filters.domains.length > 0 &&
                !filters.domains.includes(
                    node.domain
                )
            ){

                return false;

            }


            // -------------------------------------------------
            // CATEGORY
            // -------------------------------------------------

            if(
                filters.categories.length > 0
            ){

                const hasCategory =
                    node.text.categories.some(
                        category =>
                            filters.categories.includes(
                                category
                            )
                    );


                if(!hasCategory){

                    return false;

                }

            }


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
                    node => node.domain
                )
                .filter(Boolean)

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


    const categories =
        filteredNodes.flatMap(
            node =>
                node.text.categories
        );


    return Array.from(

        new Set(

            categories
                .filter(Boolean)

        )

    ).sort();

}