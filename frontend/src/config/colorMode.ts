//frontend/src/config/colorMode.ts

export type ColorMode =
    | "domain"
    | "cluster"
    | "category"
    | "feel";


export const COLOR_MODE_OPTIONS = [

    {
        id: "domain" as const,
        label: "Domain",
        description:
            "Color points according to their original domain."
    },

    {
        id: "cluster" as const,
        label: "Cluster",
        description:
            "Color points according to their semantic cluster."
    },

    {
        id: "category" as const,
        label: "Category",
        description:
            "Color points according to their main category."
    },

    {
        id: "feel" as const,
        label: "Feel",
        description:
            "Color points according to their semantic feel."
    }

];