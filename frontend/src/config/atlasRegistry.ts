// frontend/src/config/atlasRegistry.ts

export type AtlasDefinition = {

    id: string;

    label: string;

    description?: string;

    domains: string[];

};


export const ATLAS_REGISTRY: AtlasDefinition[] = [

    {
        id: "movies",
        label: "Movies",
        description: "Movie semantic atlas",
        domains: ["movies"]
    },

    {
        id: "music",
        label: "Music",
        description: "Music semantic atlas",
        domains: ["music"]
    },

    {
        id: "restaurants",
        label: "Restaurants",
        description: "Restaurant semantic atlas",
        domains: ["restaurants"]
    },

    // ========================================================
    // CROSS-DOMAIN — METHOD A
    // GENERAL SEMANTIC SPACE
    // ========================================================

    {
        id: "movies_music",
        label: "Movies + Music — General Semantic",
        description:
            "Cross-domain movie and music atlas based on general semantic similarity",
        domains: [
            "movies",
            "music"
        ]
    },

    // ========================================================
    // CROSS-DOMAIN — METHOD B
    // SHARED EXPERIENTIAL / FEEL SPACE
    // ========================================================

    {
        id: "movies_music_feel",
        label: "Movies + Music — Feel",
        description:
            "Cross-domain movie and music atlas based on shared experiential characteristics",
        domains: [
            "movies",
            "music"
        ]
    },

    // ========================================================
    // CROSS-DOMAIN — METHOD A
    // GENERAL SEMANTIC SPACE
    // ========================================================

    {
        id: "movies_music_restaurants",
        label: "Movies + Music + Restaurants — General Semantic",
        description:
            "Cross-domain movie, music and restaurant atlas based on general semantic similarity",
        domains: [
            "movies",
            "music",
            "restaurants"
        ]
    },

    // ========================================================
    // CROSS-DOMAIN — METHOD B
    // SHARED EXPERIENTIAL / FEEL SPACE
    // ========================================================

    {
        id: "movies_music_restaurants_feel",
        label: "Movies + Music + Restaurants — Feel",
        description:
            "Cross-domain movie, music and restaurant atlas based on shared experiential characteristics",
        domains: [
            "movies",
            "music",
            "restaurants"
        ]
    }

];