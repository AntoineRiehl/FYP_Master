// frontend/src/components/universe/colors.ts

import type {
    AtlasNode
} from "../../types/atlas";

import type {
    ColorMode
} from "../../config/colorMode";


// =====================================================
// PALETTE
// =====================================================

const COLORS = [

    "#ef4444",
    "#3b82f6",
    "#22c55e",
    "#eab308",
    "#a855f7",
    "#06b6d4",
    "#f97316",
    "#ec4899",
    "#84cc16",
    "#14b8a6",
    "#8b5cf6",
    "#f43f5e"

];


// =====================================================
// STABLE STRING HASH
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
                hash * 31
                +
                value.charCodeAt(i)
            )
            | 0;

    }

    return Math.abs(hash);

}


// =====================================================
// STRING → COLOR
// =====================================================

function stringColor(
    value: string
): string {

    const index =
        hashString(value)
        %
        COLORS.length;

    return COLORS[index];

}


// =====================================================
// CLUSTER COLOR
// =====================================================

export function clusterColor(
    cluster: number
): string {

    if (cluster < 0) {

        return "#6b7280";

    }

    return COLORS[
        cluster % COLORS.length
    ];

}


// =====================================================
// DOMAIN COLOR
// =====================================================

function domainColor(
    domain: string
): string {

    return stringColor(
        `domain:${domain}`
    );

}


// =====================================================
// CATEGORY COLOR
// =====================================================

function categoryColor(
    node: AtlasNode
): string {

    const category =
        node.text.categories?.[0];

    if (!category) {

        return "#6b7280";

    }

    return stringColor(
        `category:${category}`
    );

}


// =====================================================
// FEEL COLOR
// =====================================================

function feelColor(
    node: AtlasNode
): string {

    /*
     * Temporary implementation:
     *
     * Until dedicated LLM-generated feel labels
     * are available, the cluster label is used as
     * the semantic "feel" representation.
     */

    const feel =
        node.visual.cluster_label
        ??
        "Unknown";

    return stringColor(
        `feel:${feel}`
    );

}


// =====================================================
// MAIN COLOR RESOLVER
// =====================================================

export function getNodeColor(

    node: AtlasNode,

    mode: ColorMode

): string {

    switch (mode) {

        case "domain":

            return domainColor(
                node.domain
            );


        case "cluster":

            return clusterColor(
                node.visual.cluster ?? -1
            );


        case "category":

            return categoryColor(
                node
            );


        case "feel":

            return feelColor(
                node
            );


        default:

            return "#6b7280";

    }

}