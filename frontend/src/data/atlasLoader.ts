// frontend/src/data/atlasLoader.ts

import type { AtlasData } from "../types/atlas";

export async function loadAtlas(
    atlasName: string
): Promise<AtlasData> {

    const basePath = `/data/${atlasName}`;

    const [
        atlasResponse,
        landmarksResponse,
        regionsResponse
    ] = await Promise.all([

        fetch(`${basePath}/atlas.json`),

        fetch(`${basePath}/landmarks.json`),

        fetch(`${basePath}/regions.json`)
    ]);

    if (
        !atlasResponse.ok ||
        !landmarksResponse.ok ||
        !regionsResponse.ok
    ) {

        throw new Error(
            `Failed to load atlas "${atlasName}".`
        );
    }

    const [
        atlas,
        landmarks,
        regions
    ] = await Promise.all([

        atlasResponse.json(),

        landmarksResponse.json(),

        regionsResponse.json()
    ]);

    return {
        atlas,
        landmarks,
        regions
    };
}