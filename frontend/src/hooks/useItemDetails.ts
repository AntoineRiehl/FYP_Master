// frontend/src/hooks/useItemDetails.ts


import {
    useCallback,
    useEffect,
    useRef,
    useState
} from "react";


import type {

    AtlasNode,

    ItemDetailsManifest,

    ItemProfile,

    ItemReviewBundle

} from "../types/atlas";


// =====================================================
// PATHS
// =====================================================

const ITEM_DETAILS_ROOT =
    "/data/item_details";


const MANIFEST_URL =
    `${ITEM_DETAILS_ROOT}/manifest.json`;


// =====================================================
// TYPES
// =====================================================

type ProfileShard =

    Record<
        string,
        ItemProfile
    >;


type ReviewShard =

    Record<
        string,
        ItemReviewBundle
    >;


export type UseItemDetailsResult = {

    // -------------------------------------------------
    // PROFILE
    // -------------------------------------------------

    profile:
        ItemProfile | null;

    profileLoading:
        boolean;

    profileError:
        Error | null;


    // -------------------------------------------------
    // REVIEWS
    // -------------------------------------------------

    reviews:
        ItemReviewBundle | null;

    reviewsLoading:
        boolean;

    reviewsLoaded:
        boolean;

    reviewsError:
        Error | null;


    // -------------------------------------------------
    // ACTIONS
    // -------------------------------------------------

    loadReviews:
        () => Promise<void>;

};


// =====================================================
// JSON CACHE
// =====================================================

/*
 * The same shard can contain hundreds or thousands
 * of entities.
 *
 * Once the browser has loaded a shard, keep the parsed
 * JSON Promise in memory so selecting another item from
 * the same shard does not trigger another network fetch.
 */

const jsonCache =

    new Map<
        string,
        Promise<unknown>
    >();


// =====================================================
// FETCH JSON
// =====================================================

async function fetchJson<T>(

    url: string

): Promise<T> {


    const existing =

        jsonCache.get(
            url
        );


    if (existing) {

        return existing.then(

            value =>
                value as T

        );

    }


    const request:
        Promise<T> =

        fetch(
            url
        )

            .then(

                async response => {


                    if (
                        !response.ok
                    ) {

                        throw new Error(

                            `Failed to load ${url} `
                            +
                            `(${response.status} `
                            +
                            `${response.statusText}).`

                        );

                    }


                    const value:
                        unknown =

                        await response.json();


                    return value as T;

                }

            );


    /*
     * Promise<T> is safe to store as Promise<unknown>
     * because the cache intentionally does not know
     * which schema belongs to each URL.
     */

    jsonCache.set(

        url,

        request

    );


    request.catch(
        () => {

            jsonCache.delete(
                url
            );

        }
    );


    return request;

}


// =====================================================
// LOAD MANIFEST
// =====================================================

function loadManifest():

    Promise<ItemDetailsManifest> {


    return fetchJson<
        ItemDetailsManifest
    >(
        MANIFEST_URL
    );

}


// =====================================================
// SOURCE ID
// =====================================================

function getNodeSourceId(

    node: AtlasNode

): string | null {


    const value =

        node.source_id
        ??
        node.id;


    if (
        value === null
        ||
        value === undefined
    ) {

        return null;

    }


    const result =

        String(
            value
        ).trim();


    if (!result) {

        return null;

    }


    return result;

}


// =====================================================
// DOMAIN
// =====================================================

function getNodeDomain(

    node: AtlasNode

): string {


    return node.domain
        .trim()
        .toLowerCase();

}


// =====================================================
// STABLE FNV-1A HASH
// =====================================================

function fnv1a32(

    value: string

): number {


    /*
     * This reproduces the algorithm used by:
     *
     * build_frontend_item_details.py
     *
     * Python:
     *
     * hash = 2166136261
     *
     * for character in value:
     *     hash ^= ord(character)
     *     hash = (
     *         hash * 16777619
     *     ) & 0xFFFFFFFF
     *
     * Math.imul() gives us deterministic 32-bit
     * integer multiplication in JavaScript.
     */

    let hash =
        2166136261;


    for (
        const character
        of value
    ) {


        const codePoint =

            character.codePointAt(
                0
            )

            ??

            0;


        hash ^=
            codePoint;


        hash =

            Math.imul(

                hash,

                16777619

            )

            >>>

            0;

    }


    return hash >>> 0;

}


// =====================================================
// SHARD INDEX
// =====================================================

function getShardIndex(

    sourceId: string,

    shardCount: number

): number {


    return (

        fnv1a32(
            sourceId
        )

        %

        shardCount

    );

}


// =====================================================
// SHARD FILE NAME
// =====================================================

function getShardFilename(

    shardIndex: number

): string {


    return (

        "shard_"

        +

        String(
            shardIndex
        )
            .padStart(
                3,
                "0"
            )

        +

        ".json"

    );

}


// =====================================================
// PROFILE URL
// =====================================================

function getProfileShardUrl(

    domain: string,

    shardIndex: number

): string {


    return (

        `${ITEM_DETAILS_ROOT}`
        +
        `/profiles/${domain}`
        +
        `/${getShardFilename(
            shardIndex
        )}`

    );

}


// =====================================================
// REVIEW URL
// =====================================================

function getReviewShardUrl(

    domain: string,

    shardIndex: number

): string {


    return (

        `${ITEM_DETAILS_ROOT}`
        +
        `/reviews/${domain}`
        +
        `/${getShardFilename(
            shardIndex
        )}`

    );

}


// =====================================================
// SELECTION KEY
// =====================================================

function getSelectionKey(

    node: AtlasNode | null

): string | null {


    if (!node) {

        return null;

    }


    const sourceId =

        getNodeSourceId(
            node
        );


    if (!sourceId) {

        return null;

    }


    return (

        getNodeDomain(
            node
        )

        +

        ":"

        +

        sourceId

    );

}


// =====================================================
// EMPTY REVIEW BUNDLE
// =====================================================

function createEmptyReviewBundle():

    ItemReviewBundle {


    return {

        available_count:
            0,

        sampled_count:
            0,

        reviews:
            []

    };

}


// =====================================================
// HOOK
// =====================================================

export function useItemDetails(

    node: AtlasNode | null

): UseItemDetailsResult {


    // =================================================
    // PROFILE STATE
    // =====================================================

    const [
        profile,
        setProfile
    ] =
        useState<ItemProfile | null>(
            null
        );


    const [
        profileLoading,
        setProfileLoading
    ] =
        useState(false);


    const [
        profileError,
        setProfileError
    ] =
        useState<Error | null>(
            null
        );


    // =================================================
    // REVIEW STATE
    // =====================================================

    const [
        reviews,
        setReviews
    ] =
        useState<ItemReviewBundle | null>(
            null
        );


    const [
        reviewsLoading,
        setReviewsLoading
    ] =
        useState(false);


    const [
        reviewsLoaded,
        setReviewsLoaded
    ] =
        useState(false);


    const [
        reviewsError,
        setReviewsError
    ] =
        useState<Error | null>(
            null
        );


    // =================================================
    // CURRENT SELECTION KEY
    // =====================================================

    /*
     * This protects us against stale asynchronous
     * responses.
     *
     * Example:
     *
     * select Toy Story
     *      ↓
     * profile fetch begins
     *      ↓
     * immediately select Jumanji
     *      ↓
     * Toy Story request finishes later
     *
     * The old response must NOT overwrite Jumanji.
     */

    const selectionKeyRef =

        useRef<string | null>(
            null
        );


    const selectionKey =

        getSelectionKey(
            node
        );


    selectionKeyRef.current =
        selectionKey;


    // =================================================
    // LOAD PROFILE WHEN SELECTION CHANGES
    // =====================================================

    useEffect(() => {


        // -------------------------------------------------
        // RESET ALL ITEM-SPECIFIC STATE
        // -------------------------------------------------

        setProfile(
            null
        );


        setProfileError(
            null
        );


        setReviews(
            null
        );


        setReviewsLoading(
            false
        );


        setReviewsLoaded(
            false
        );


        setReviewsError(
            null
        );


        // -------------------------------------------------
        // NO SELECTION
        // -------------------------------------------------

        if (
            !node
            ||
            !selectionKey
        ) {


            setProfileLoading(
                false
            );


            return;

        }


        const expectedSelectionKey =

            selectionKey;


        let cancelled =
            false;


        // =================================================
        // LOAD SELECTED PROFILE
        // =====================================================

        async function loadProfile() {


            setProfileLoading(
                true
            );


            try {


                // -----------------------------------------
                // SOURCE ID
                // -----------------------------------------

                const sourceId =

                    getNodeSourceId(
                        node!
                    );


                if (!sourceId) {

                    throw new Error(

                        "Selected item does not have "
                        +
                        "a usable source ID."

                    );

                }


                // -----------------------------------------
                // DOMAIN
                // -----------------------------------------

                const domain =

                    getNodeDomain(
                        node!
                    );


                // -----------------------------------------
                // MANIFEST
                // -----------------------------------------

                const manifest =

                    await loadManifest();


                // -----------------------------------------
                // SHARD
                // -----------------------------------------

                const shardIndex =

                    getShardIndex(

                        sourceId,

                        manifest
                            .sharding
                            .shard_count

                    );


                const url =

                    getProfileShardUrl(

                        domain,

                        shardIndex

                    );


                const shard =

                    await fetchJson<
                        ProfileShard
                    >(
                        url
                    );


                // -----------------------------------------
                // STALE REQUEST CHECK
                // -----------------------------------------

                if (
                    cancelled

                    ||

                    selectionKeyRef.current
                    !==
                    expectedSelectionKey
                ) {

                    return;

                }


                // -----------------------------------------
                // ENTITY PROFILE
                // -----------------------------------------

                const itemProfile =

                    shard[
                        sourceId
                    ];


                if (!itemProfile) {

                    throw new Error(

                        "No frontend item profile found "
                        +
                        `for ${domain}:${sourceId}.`

                    );

                }


                setProfile(
                    itemProfile
                );

            }


            catch (error) {


                if (
                    cancelled

                    ||

                    selectionKeyRef.current
                    !==
                    expectedSelectionKey
                ) {

                    return;

                }


                const errorObject =

                    error instanceof Error

                    ?

                    error

                    :

                    new Error(

                        "Failed to load item profile."

                    );


                console.error(

                    "Failed to load item profile:",

                    errorObject

                );


                setProfileError(
                    errorObject
                );

            }


            finally {


                if (

                    !cancelled

                    &&

                    selectionKeyRef.current
                    ===
                    expectedSelectionKey

                ) {

                    setProfileLoading(
                        false
                    );

                }

            }

        }


        loadProfile();


        // =================================================
        // CLEANUP
        // =====================================================

        return () => {

            cancelled =
                true;

        };


    }, [
        node,
        selectionKey
    ]);


    // =================================================
    // LAZY LOAD REVIEWS
    // =====================================================

    const loadReviews =

        useCallback(

            async () => {


                // -----------------------------------------
                // NO ITEM
                // -----------------------------------------

                if (
                    !node
                    ||
                    !selectionKey
                ) {

                    return;

                }


                // -----------------------------------------
                // ALREADY LOADED / LOADING
                // -----------------------------------------

                if (
                    reviewsLoaded
                    ||
                    reviewsLoading
                ) {

                    return;

                }


                const expectedSelectionKey =

                    selectionKey;


                // -----------------------------------------
                // SOURCE ID
                // -----------------------------------------

                const sourceId =

                    getNodeSourceId(
                        node
                    );


                if (!sourceId) {


                    setReviewsError(

                        new Error(

                            "Selected item does not have "
                            +
                            "a usable source ID."

                        )

                    );


                    return;

                }


                // -----------------------------------------
                // START
                // -----------------------------------------

                setReviewsLoading(
                    true
                );


                setReviewsError(
                    null
                );


                try {


                    // -------------------------------------
                    // PROFILE SAYS THERE ARE NO REVIEWS
                    // -------------------------------------

                    if (

                        profile

                        &&

                        profile
                            .reviews
                            .available_count === 0

                    ) {


                        if (

                            selectionKeyRef.current
                            ===
                            expectedSelectionKey

                        ) {


                            setReviews(

                                createEmptyReviewBundle()

                            );


                            setReviewsLoaded(
                                true
                            );

                        }


                        return;

                    }


                    // -------------------------------------
                    // DOMAIN
                    // -------------------------------------

                    const domain =

                        getNodeDomain(
                            node
                        );


                    // -------------------------------------
                    // MANIFEST
                    // -------------------------------------

                    const manifest =

                        await loadManifest();


                    // -------------------------------------
                    // REVIEW SHARD
                    // -------------------------------------

                    const shardIndex =

                        getShardIndex(

                            sourceId,

                            manifest
                                .sharding
                                .shard_count

                        );


                    const url =

                        getReviewShardUrl(

                            domain,

                            shardIndex

                        );


                    const shard =

                        await fetchJson<
                            ReviewShard
                        >(
                            url
                        );


                    // -------------------------------------
                    // STALE REQUEST CHECK
                    // -------------------------------------

                    if (

                        selectionKeyRef.current
                        !==
                        expectedSelectionKey

                    ) {

                        return;

                    }


                    // -------------------------------------
                    // REVIEW BUNDLE
                    // -------------------------------------

                    const itemReviews =

                        shard[
                            sourceId
                        ]

                        ??

                        createEmptyReviewBundle();


                    setReviews(
                        itemReviews
                    );


                    setReviewsLoaded(
                        true
                    );

                }


                catch (error) {


                    if (

                        selectionKeyRef.current
                        !==
                        expectedSelectionKey

                    ) {

                        return;

                    }


                    const errorObject =

                        error instanceof Error

                        ?

                        error

                        :

                        new Error(

                            "Failed to load item reviews."

                        );


                    console.error(

                        "Failed to load item reviews:",

                        errorObject

                    );


                    setReviewsError(
                        errorObject
                    );

                }


                finally {


                    if (

                        selectionKeyRef.current
                        ===
                        expectedSelectionKey

                    ) {

                        setReviewsLoading(
                            false
                        );

                    }

                }

            },

            [
                node,
                selectionKey,
                profile,
                reviewsLoaded,
                reviewsLoading
            ]

        );


    // =================================================
    // RESULT
    // =====================================================

    return {

        profile,

        profileLoading,

        profileError,


        reviews,

        reviewsLoading,

        reviewsLoaded,

        reviewsError,


        loadReviews

    };

}