//frontend/src/config/atlasRegistry.ts

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

    {
        id: "movies_music",
        label: "Movies + Music",
        description: "Cross-domain movie and music semantic atlas",
        domains: [
            "movies",
            "music"
        ]
    },

    {
        id: "movies_music_restaurants",
        label: "Movies + Music + Restaurants",
        description:
            "Cross-domain movie, music and restaurant semantic atlas",
        domains: [
            "movies",
            "music",
            "restaurants"
        ]
    }

];