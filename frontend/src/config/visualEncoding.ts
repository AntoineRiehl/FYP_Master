//frontend/src/config/visualEncoding.ts

import type {
    ColorMode
} from "../types/atlas";


// =====================================================
// COLOR MODE DEFINITIONS
// =====================================================

export type ColorModeDefinition = {

    id: ColorMode;

    label: string;

    description: string;

};


// =====================================================
// AVAILABLE COLOR MODES
// =====================================================

export const COLOR_MODES: ColorModeDefinition[] = [

    {
        id: "domain",

        label: "Domain",

        description:
            "Colour items according to their original domain."

    },

    {
        id: "cluster",

        label: "Cluster",

        description:
            "Colour items according to their semantic cluster."

    },

    {
        id: "category",

        label: "Category",

        description:
            "Colour items according to their domain-specific category."

    }

];


// =====================================================
// DEFAULT
// =====================================================

export const DEFAULT_COLOR_MODE: ColorMode =
    "domain";