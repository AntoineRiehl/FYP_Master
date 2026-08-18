//frontend/scr/hooks/useAtlas.ts

import { useEffect, useState } from "react";

import { loadAtlas } from "../data/atlasLoader";

import type {
    AtlasData
} from "../types/atlas";


// =====================================================
// RETURN TYPE
// =====================================================

type UseAtlasResult = {

    data: AtlasData | null;

    loading: boolean;

    error: Error | null;

};


// =====================================================
// HOOK
// =====================================================

export function useAtlas(
    atlasName: string
): UseAtlasResult {

    const [
        data,
        setData
    ] =
        useState<AtlasData | null>(null);


    const [
        loading,
        setLoading
    ] =
        useState(true);


    const [
        error,
        setError
    ] =
        useState<Error | null>(null);


    useEffect(() => {

        let cancelled = false;


        async function loadSelectedAtlas() {

            // -------------------------------------------------
            // RESET STATE
            // -------------------------------------------------

            setLoading(true);

            setError(null);

            setData(null);


            try {

                // -------------------------------------------------
                // LOAD ATLAS
                // -------------------------------------------------

                const atlas =
                    await loadAtlas(
                        atlasName
                    );


                // -------------------------------------------------
                // PREVENT STALE REQUESTS
                // -------------------------------------------------

                if (cancelled) {

                    return;

                }


                // -------------------------------------------------
                // STORE DATA
                // -------------------------------------------------

                setData(atlas);

            }


            catch (err) {

                if (cancelled) {

                    return;

                }


                const errorObject =
                    err instanceof Error

                        ? err

                        : new Error(
                            "Failed to load atlas."
                        );


                console.error(
                    `Failed to load atlas "${atlasName}":`,
                    errorObject
                );


                setError(
                    errorObject
                );

            }


            finally {

                if (!cancelled) {

                    setLoading(false);

                }

            }

        }


        loadSelectedAtlas();


        // -------------------------------------------------
        // CLEANUP
        // -------------------------------------------------

        return () => {

            cancelled = true;

        };

    }, [atlasName]);


    return {

        data,

        loading,

        error

    };

}