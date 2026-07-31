// frontend/src/hooks/useAtlas.ts

import { useEffect, useState } from "react";
import { loadAtlas } from "../data/atlasLoader";
import type { AtlasData } from "../types/atlas";

export function useAtlas(atlasName: string) {

    const [data, setData] =
        useState<AtlasData | null>(null);

    useEffect(() => {
        loadAtlas(atlasName)
            .then(setData);
    }, [atlasName]);

    return data;
}