// frontend/src/components/DetailsPanel.tsx

import {
    useEffect,
    useMemo,
    useState
} from "react";


import type {
    AtlasNode,
    FeelProfile,
    ItemDetailReview
} from "../types/atlas";


import type {
    NeighborAnalysis,
    NeighborResult
} from "./universe/neighbors";


import type {
    NodeClusterStats
} from "../utils/clusterStats";


import {
    useItemDetails
} from "../hooks/useItemDetails";


import {
    ATLAS_REGISTRY
} from "../config/atlasRegistry";


// =====================================================
// PROPS
// =====================================================

type Props = {

    node:
        AtlasNode | null;

    atlasName:
        string;

    neighborAnalysis:
        NeighborAnalysis | null;

    clusterStats:
        NodeClusterStats | null;

    onSelectNode:
        (node: AtlasNode) => void;

};


// =====================================================
// FEEL DIMENSIONS
// =====================================================

type FeelDimension =
    keyof FeelProfile;


const FEEL_DIMENSIONS:
    FeelDimension[] = [

        "valence",
        "activation",
        "potency",
        "tension",
        "warmth",
        "scale",
        "tone",
        "familiarity",
        "refinement",
        "complexity",
        "nostalgia",
        "wonder",
        "tenderness"

    ];


// =====================================================
// BIPOLAR LABELS
// =====================================================

const BIPOLAR_LABELS:

    Partial<
        Record<
            FeelDimension,
            {
                positive: string;
                negative: string;
            }
        >
    >

    = {

        valence: {
            positive: "Positive",
            negative: "Negative"
        },

        activation: {
            positive: "Energetic",
            negative: "Calm"
        },

        potency: {
            positive: "Powerful",
            negative: "Gentle"
        },

        tension: {
            positive: "Unsettling",
            negative: "Comforting"
        },

        warmth: {
            positive: "Warm",
            negative: "Cold"
        },

        scale: {
            positive: "Grand",
            negative: "Intimate"
        },

        tone: {
            positive: "Playful",
            negative: "Serious"
        },

        familiarity: {
            positive: "Familiar",
            negative: "Novel"
        },

        refinement: {
            positive: "Refined",
            negative: "Raw"
        },

        complexity: {
            positive: "Complex",
            negative: "Simple"
        }

    };


// =====================================================
// UNIPOLAR LABELS
// =====================================================

const UNIPOLAR_LABELS:

    Partial<
        Record<
            FeelDimension,
            string
        >
    >

    = {

        nostalgia:
            "Nostalgia",

        wonder:
            "Wonder",

        tenderness:
            "Tenderness"

    };


// =====================================================
// DISPLAY HELPERS
// =====================================================

function domainLabel(
    domain: string
): string {


    switch (
        domain.toLowerCase()
    ) {

        case "movies":
            return "Movie";

        case "music":
            return "Music";

        case "restaurants":
            return "Restaurant";

        default:
            return domain;

    }

}


function formatNumber(
    value:
        number
        |
        null
        |
        undefined
): string {


    if (
        value === null
        ||
        value === undefined
        ||
        !Number.isFinite(value)
    ) {

        return "—";

    }


    return Math.round(
        value
    ).toLocaleString();

}


function formatTopPercent(
    value:
        number
        |
        null
        |
        undefined
): string {


    if (
        value === null
        ||
        value === undefined
        ||
        !Number.isFinite(value)
    ) {

        return "Unavailable";

    }


    if (
        value < 1
    ) {

        return "Top <1%";

    }


    return (
        `Top ${Math.max(
            1,
            Math.round(value)
        )}%`
    );

}


function formatSignedScore(
    value: number
): string {


    const prefix =
        value >= 0
        ?
        "+"
        :
        "";


    return (
        `${prefix}${value.toFixed(2)}σ`
    );

}


// =====================================================
// FEEL DESCRIPTOR
// =====================================================

function getFeelDescriptor(

    dimension: FeelDimension,

    value: number

): string {


    const bipolar =
        BIPOLAR_LABELS[
            dimension
        ];


    if (bipolar) {

        return (
            value >= 0
            ?
            bipolar.positive
            :
            bipolar.negative
        );

    }


    const unipolar =
        UNIPOLAR_LABELS[
            dimension
        ];


    if (unipolar) {

        if (
            value >= 0
        ) {

            return unipolar;

        }


        return (
            `Low ${unipolar.toLowerCase()}`
        );

    }


    return dimension;

}


// =====================================================
// CATEGORY CLEANING
// =====================================================

function cleanCategories(
    node: AtlasNode
): string[] {


    const categories =
        node.text.categories
        ??
        [];


    const cleaned =

        categories

            .flatMap(

                category =>
                    category.split("|")

            )

            .map(
                category =>
                    category.trim()
            )

            .filter(
                Boolean
            );


    return Array.from(
        new Set(cleaned)
    );

}


// =====================================================
// ATLAS CONTEXT
// =====================================================

function getAtlasContext(
    atlasName: string
) {


    const definition =
        ATLAS_REGISTRY.find(

            atlas =>
                atlas.id === atlasName

        );


    const domains =
        definition?.domains
        ??
        [];


    const isFeelAtlas =
        atlasName.includes(
            "_feel"
        );


    let basis =
        "Mono-domain Semantic";


    if (isFeelAtlas) {

        basis =
            "Shared Experiential / Feel";

    }

    else if (
        domains.length > 1
    ) {

        basis =
            "General Semantic";

    }


    return {

        label:
            definition?.label
            ??
            atlasName,

        domains,

        isFeelAtlas,

        basis

    };

}


// =====================================================
// SECTION
// =====================================================

function Section({

    title,

    children

}: {

    title: string;

    children:
        React.ReactNode;

}) {


    return (

        <section
            style={{

                paddingTop:
                    "18px",

                paddingBottom:
                    "18px",

                borderTop:
                    "1px solid rgba(255,255,255,0.09)"

            }}
        >

            <div
                style={{

                    marginBottom:
                        "12px",

                    fontSize:
                        "11px",

                    fontWeight:
                        700,

                    letterSpacing:
                        "0.09em",

                    color:
                        "rgba(255,255,255,0.52)"

                }}
            >
                {title.toUpperCase()}
            </div>


            {children}

        </section>

    );

}


// =====================================================
// STAT ROW
// =====================================================

function StatRow({

    label,

    value

}: {

    label: string;

    value:
        React.ReactNode;

}) {


    return (

        <div
            style={{

                display:
                    "flex",

                justifyContent:
                    "space-between",

                gap:
                    "12px",

                marginBottom:
                    "8px",

                fontSize:
                    "13px"

            }}
        >

            <span
                style={{
                    opacity: 0.58
                }}
            >
                {label}
            </span>


            <span
                style={{

                    textAlign:
                        "right",

                    fontWeight:
                        500

                }}
            >
                {value}
            </span>

        </div>

    );

}


// =====================================================
// CLICKABLE NODE
// =====================================================

function ClickableNode({

    result,

    index,

    onSelectNode

}: {

    result: NeighborResult;

    index?: number;

    onSelectNode:
        (node: AtlasNode) => void;

}) {


    return (

        <button
            type="button"

            onClick={() =>
                onSelectNode(
                    result.node
                )
            }

            style={{

                width:
                    "100%",

                display:
                    "flex",

                alignItems:
                    "center",

                gap:
                    "9px",

                marginBottom:
                    "6px",

                padding:
                    "8px 9px",

                border:
                    "1px solid rgba(255,255,255,0.07)",

                borderRadius:
                    "6px",

                background:
                    "rgba(255,255,255,0.025)",

                color:
                    "white",

                cursor:
                    "pointer",

                textAlign:
                    "left"

            }}
        >

            {
                index !== undefined
                &&
                (
                    <span
                        style={{

                            width:
                                "18px",

                            opacity:
                                0.38,

                            fontSize:
                                "11px"

                        }}
                    >
                        {index}
                    </span>
                )
            }


            <div
                style={{

                    minWidth:
                        0,

                    flex:
                        1

                }}
            >

                <div
                    style={{

                        whiteSpace:
                            "nowrap",

                        overflow:
                            "hidden",

                        textOverflow:
                            "ellipsis",

                        fontSize:
                            "13px",

                        fontWeight:
                            500

                    }}
                >
                    {result.node.title}
                </div>


                <div
                    style={{

                        marginTop:
                            "2px",

                        fontSize:
                            "10px",

                        opacity:
                            0.45

                    }}
                >
                    {
                        domainLabel(
                            result.node.domain
                        )
                    }
                </div>

            </div>


            <span
                style={{

                    opacity:
                        0.32,

                    fontSize:
                        "15px"

                }}
            >
                ›
            </span>

        </button>

    );

}


// =====================================================
// FEEL ROW
// =====================================================

function FeelRow({

    dimension,

    value

}: {

    dimension: FeelDimension;

    value: number;

}) {


    const descriptor =
        getFeelDescriptor(

            dimension,

            value

        );


    const width =

        Math.min(

            100,

            Math.abs(value)
            /
            3
            *
            100

        );


    return (

        <div
            style={{
                marginBottom: "10px"
            }}
        >

            <div
                style={{

                    display:
                        "flex",

                    justifyContent:
                        "space-between",

                    gap:
                        "10px",

                    marginBottom:
                        "4px",

                    fontSize:
                        "12px"

                }}
            >

                <span>
                    {descriptor}
                </span>


                <span
                    style={{
                        opacity: 0.55
                    }}
                >
                    {
                        formatSignedScore(
                            value
                        )
                    }
                </span>

            </div>


            <div
                style={{

                    height:
                        "4px",

                    borderRadius:
                        "999px",

                    overflow:
                        "hidden",

                    background:
                        "rgba(255,255,255,0.08)"

                }}
            >

                <div
                    style={{

                        width:
                            `${width}%`,

                        height:
                            "100%",

                        borderRadius:
                            "999px",

                        background:
                            "rgba(255,255,255,0.65)"

                    }}
                />

            </div>

        </div>

    );

}


// =====================================================
// REVIEW VIEWER
// =====================================================

function ReviewViewer({

    review,

    index,

    count,

    onPrevious,

    onNext

}: {

    review: ItemDetailReview;

    index: number;

    count: number;

    onPrevious:
        () => void;

    onNext:
        () => void;

}) {


    return (

        <div>

            <div
                style={{

                    display:
                        "flex",

                    justifyContent:
                        "space-between",

                    marginBottom:
                        "10px",

                    fontSize:
                        "11px",

                    opacity:
                        0.55

                }}
            >

                <span>
                    Review {index + 1} / {count}
                </span>


                <span>

                    {
                        review.rating !== null
                        &&
                        review.rating !== undefined
                        &&
                        `Rating ${review.rating}`
                    }

                </span>

            </div>


            <div
                style={{

                    maxHeight:
                        "220px",

                    overflowY:
                        "auto",

                    padding:
                        "12px",

                    borderRadius:
                        "6px",

                    background:
                        "rgba(255,255,255,0.035)",

                    fontSize:
                        "12px",

                    lineHeight:
                        1.55,

                    whiteSpace:
                        "pre-wrap"

                }}
            >

                {review.text}

            </div>


            {
                (
                    review.source
                    ||
                    review.date
                )
                &&
                (
                    <div
                        style={{

                            marginTop:
                                "7px",

                            fontSize:
                                "10px",

                            opacity:
                                0.42

                        }}
                    >

                        {
                            [
                                review.source,
                                review.date
                            ]
                                .filter(
                                    Boolean
                                )
                                .join(" · ")
                        }

                    </div>
                )
            }


            {
                count > 1
                &&
                (
                    <div
                        style={{

                            display:
                                "flex",

                            justifyContent:
                                "space-between",

                            gap:
                                "8px",

                            marginTop:
                                "10px"

                        }}
                    >

                        <button
                            type="button"

                            onClick={
                                onPrevious
                            }

                            style={{
                                flex: 1
                            }}
                        >
                            ← Previous
                        </button>


                        <button
                            type="button"

                            onClick={
                                onNext
                            }

                            style={{
                                flex: 1
                            }}
                        >
                            Next →
                        </button>

                    </div>
                )
            }

        </div>

    );

}


// =====================================================
// COMPONENT
// =====================================================

export default function DetailsPanel({

    node,

    atlasName,

    neighborAnalysis,

    clusterStats,

    onSelectNode

}: Props) {


    // =================================================
    // ITEM DETAIL DATA
    // =====================================================

    const {

        profile,

        profileLoading,

        profileError,

        reviews,

        reviewsLoading,

        reviewsLoaded,

        reviewsError,

        loadReviews

    } =
        useItemDetails(
            node
        );


    // =================================================
    // LOCAL UI STATE
    // =====================================================

    const [
        showAllFeel,
        setShowAllFeel
    ] =
        useState(false);


    const [
        moreExpanded,
        setMoreExpanded
    ] =
        useState(false);


    const [
        reviewIndex,
        setReviewIndex
    ] =
        useState(0);


    // =================================================
    // RESET EXPANSIONS ON SELECTION
    // =====================================================

    useEffect(() => {

        setShowAllFeel(
            false
        );

        setMoreExpanded(
            false
        );

        setReviewIndex(
            0
        );

    }, [
        node?.id
    ]);


    // =================================================
    // KEEP REVIEW INDEX VALID
    // =====================================================

    useEffect(() => {

        if (
            !reviews
            ||
            reviews.reviews.length === 0
        ) {

            setReviewIndex(
                0
            );

            return;

        }


        if (
            reviewIndex
            >=
            reviews.reviews.length
        ) {

            setReviewIndex(
                0
            );

        }

    }, [
        reviews,
        reviewIndex
    ]);


    // =================================================
    // ATLAS CONTEXT
    // =====================================================

    const atlasContext =
        useMemo(

            () =>
                getAtlasContext(
                    atlasName
                ),

            [
                atlasName
            ]

        );


    // =================================================
    // FEEL ENTRIES
    // =====================================================

    const feelEntries =
        useMemo(

            () => {

                if (
                    !profile?.feel
                ) {

                    return [];

                }


                return FEEL_DIMENSIONS.map(

                    dimension => ({

                        dimension,

                        value:
                            profile.feel![
                                dimension
                            ]

                    })

                );

            },

            [
                profile
            ]

        );


    const strongestFeelEntries =
        useMemo(

            () =>
                [...feelEntries]

                    .sort(

                        (a, b) =>

                            Math.abs(
                                b.value
                            )
                            -
                            Math.abs(
                                a.value
                            )

                    )

                    .slice(
                        0,
                        5
                    ),

            [
                feelEntries
            ]

        );


    // =================================================
    // CLEAN CATEGORIES
    // =====================================================

    const categories =
        useMemo(

            () =>
                node
                ?
                cleanCategories(
                    node
                )
                :
                [],

            [
                node
            ]

        );


    // =================================================
    // EXPAND MORE INFORMATION
    // =====================================================

    function handleToggleMore() {


        const next =
            !moreExpanded;


        setMoreExpanded(
            next
        );


        if (
            next
            &&
            !reviewsLoaded
            &&
            !reviewsLoading
        ) {

            void loadReviews();

        }

    }


    // =================================================
    // EMPTY STATE
    // =====================================================

    if (!node) {

        return (

            <aside
                style={{

                    width:
                        "380px",

                    padding:
                        "22px",

                    background:
                        "#0b1020",

                    borderLeft:
                        "1px solid rgba(255,255,255,0.1)",

                    boxSizing:
                        "border-box",

                    overflowY:
                        "auto"

                }}
            >

                <h2
                    style={{

                        marginTop:
                            0,

                        fontSize:
                            "20px"

                    }}
                >
                    Details
                </h2>


                <div
                    style={{

                        marginTop:
                            "30px",

                        opacity:
                            0.5,

                        fontSize:
                            "14px",

                        lineHeight:
                            1.5

                    }}
                >
                    Select an item to inspect its
                    neighbourhood, experiential profile
                    and position within the atlas.
                </div>

            </aside>

        );

    }


    // =================================================
    // REVIEW COUNT
    // =====================================================

    const reviewCount =

        profile?.reviews.available_count

        ??

        node.enrichment?.review_count

        ??

        0;


    // =================================================
    // DOMAIN LINKS
    // =====================================================

    const domainLinks:

        {
            domain: string;
            result: NeighborResult;
        }[] = [];


    if (
        atlasContext.domains.length > 1
        &&
        neighborAnalysis
    ) {


        for (
            const domain
            of atlasContext.domains
        ) {


            const result =

                domain === node.domain

                ?

                neighborAnalysis
                    .closestSameDomain

                :

                neighborAnalysis
                    .closestByDomain[
                        domain
                    ];


            if (result) {

                domainLinks.push({

                    domain,

                    result

                });

            }

        }

    }


    // =================================================
    // RENDER
    // =====================================================

    return (

        <aside
            style={{

                width:
                    "380px",

                padding:
                    "22px",

                background:
                    "#0b1020",

                borderLeft:
                    "1px solid rgba(255,255,255,0.1)",

                boxSizing:
                    "border-box",

                overflowY:
                    "auto"

            }}
        >


            {/* =================================================
                IDENTITY
            ================================================= */}

            <div
                style={{
                    paddingBottom: "18px"
                }}
            >

                <div
                    style={{

                        marginBottom:
                            "6px",

                        fontSize:
                            "10px",

                        fontWeight:
                            700,

                        letterSpacing:
                            "0.1em",

                        opacity:
                            0.42

                    }}
                >
                    {
                        domainLabel(
                            node.domain
                        ).toUpperCase()
                    }
                </div>


                <h2
                    style={{

                        margin:
                            0,

                        fontSize:
                            "21px",

                        lineHeight:
                            1.25,
                        
                        color: "white"

                    }}
                >
                    {node.title}
                </h2>


                <div
                    style={{

                        marginTop:
                            "10px",

                        fontSize:
                            "12px",

                        lineHeight:
                            1.6,

                        opacity:
                            0.65

                    }}
                >

                    {
                        node.metadata.year !== null
                        &&
                        node.metadata.year !== undefined
                        &&
                        (
                            <div>
                                Year: {node.metadata.year}
                            </div>
                        )
                    }


                    {
                        node.metadata.artist
                        &&
                        node.metadata.artist !== node.title
                        &&
                        (
                            <div>
                                Artist: {node.metadata.artist}
                            </div>
                        )
                    }


                    {
                        node.metadata.director
                        &&
                        (
                            <div>
                                Director: {node.metadata.director}
                            </div>
                        )
                    }


                    {
                        node.metadata.city
                        &&
                        (
                            <div>
                                Location: {node.metadata.city}
                            </div>
                        )
                    }

                </div>


                <div
                    style={{

                        display:
                            "grid",

                        gridTemplateColumns:
                            "1fr 1fr",

                        gap:
                            "8px",

                        marginTop:
                            "14px"

                    }}
                >

                    {
                        node.statistics.rating !== null
                        &&
                        node.statistics.rating !== undefined
                        &&
                        (
                            <div
                                style={{

                                    padding:
                                        "9px",

                                    borderRadius:
                                        "6px",

                                    background:
                                        "rgba(255,255,255,0.035)"

                                }}
                            >

                                <div
                                    style={{
                                        fontSize: "10px",
                                        opacity: 0.45
                                    }}
                                >
                                    RATING
                                </div>

                                <div
                                    style={{
                                        marginTop: "3px",
                                        fontWeight: 600
                                    }}
                                >
                                    {
                                        node.statistics
                                            .rating
                                            .toFixed(2)
                                    }
                                </div>

                            </div>
                        )
                    }


                    {
                        node.statistics.rating_count !== null
                        &&
                        node.statistics.rating_count !== undefined
                        &&
                        (
                            <div
                                style={{

                                    padding:
                                        "9px",

                                    borderRadius:
                                        "6px",

                                    background:
                                        "rgba(255,255,255,0.035)"

                                }}
                            >

                                <div
                                    style={{
                                        fontSize: "10px",
                                        opacity: 0.45
                                    }}
                                >
                                    RATING COUNT
                                </div>

                                <div
                                    style={{
                                        marginTop: "3px",
                                        fontWeight: 600
                                    }}
                                >
                                    {
                                        formatNumber(
                                            node.statistics
                                                .rating_count
                                        )
                                    }
                                </div>

                            </div>
                        )
                    }

                </div>


                {
                    reviewCount > 0
                    &&
                    (
                        <div
                            style={{

                                marginTop:
                                    "10px",

                                fontSize:
                                    "11px",

                                opacity:
                                    0.48

                            }}
                        >
                            {
                                formatNumber(
                                    reviewCount
                                )
                            } reviews available
                        </div>
                    )
                }

            </div>


            {/* =================================================
                ATLAS CONTEXT
            ================================================= */}

            <Section
                title="Atlas"
            >

                <div
                    style={{

                        fontSize:
                            "13px",

                        fontWeight:
                            600

                    }}
                >
                    {atlasContext.label}
                </div>


                <div
                    style={{

                        marginTop:
                            "5px",

                        fontSize:
                            "12px",

                        opacity:
                            0.55

                    }}
                >
                    {atlasContext.basis}
                </div>

            </Section>


            {/* =================================================
                NEAREST ON ATLAS
            ================================================= */}

            {
                neighborAnalysis
                &&
                neighborAnalysis.nearest.length > 0
                &&
                (
                    <Section
                        title="Nearest on atlas"
                    >

                        {
                            neighborAnalysis.nearest.map(

                                (
                                    result,
                                    index
                                ) => (

                                    <ClickableNode

                                        key={
                                            result.node.id
                                        }

                                        result={
                                            result
                                        }

                                        index={
                                            index + 1
                                        }

                                        onSelectNode={
                                            onSelectNode
                                        }

                                    />

                                )

                            )
                        }

                    </Section>
                )
            }


            {/* =================================================
                CLOSEST BY DOMAIN
            ================================================= */}

            {
                domainLinks.length > 0
                &&
                (
                    <Section
                        title="Closest by domain"
                    >

                        {
                            domainLinks.map(

                                ({
                                    domain,
                                    result
                                }) => (

                                    <div
                                        key={
                                            domain
                                        }

                                        style={{
                                            marginBottom: "11px"
                                        }}
                                    >

                                        <div
                                            style={{

                                                marginBottom:
                                                    "4px",

                                                fontSize:
                                                    "10px",

                                                opacity:
                                                    0.42

                                            }}
                                        >
                                            {
                                                domainLabel(
                                                    domain
                                                )
                                            }
                                        </div>


                                        <ClickableNode

                                            result={
                                                result
                                            }

                                            onSelectNode={
                                                onSelectNode
                                            }

                                        />

                                    </div>

                                )

                            )
                        }

                    </Section>
                )
            }


            {/* =================================================
                FEEL PROFILE
            ================================================= */}

            <Section
                title="Experiential profile"
            >

                {
                    profileLoading
                    &&
                    (
                        <div
                            style={{
                                fontSize: "12px",
                                opacity: 0.45
                            }}
                        >
                            Loading profile...
                        </div>
                    )
                }


                {
                    profileError
                    &&
                    (
                        <div
                            style={{
                                fontSize: "12px",
                                opacity: 0.55
                            }}
                        >
                            Experiential profile unavailable.
                        </div>
                    )
                }


                {
                    profile
                    &&
                    !profile.feel_defined
                    &&
                    (
                        <div
                            style={{

                                fontSize:
                                    "12px",

                                opacity:
                                    0.52,

                                lineHeight:
                                    1.5

                            }}
                        >
                            Experiential profile unavailable
                            for this item because no usable
                            semantic representation was
                            available.
                        </div>
                    )
                }


                {
                    profile?.feel
                    &&
                    (
                        <>

                            {
                                (
                                    showAllFeel
                                    ?
                                    feelEntries
                                    :
                                    strongestFeelEntries
                                )
                                .map(

                                    entry => (

                                        <FeelRow

                                            key={
                                                entry.dimension
                                            }

                                            dimension={
                                                entry.dimension
                                            }

                                            value={
                                                entry.value
                                            }

                                        />

                                    )

                                )
                            }


                            <button
                                type="button"

                                onClick={() =>
                                    setShowAllFeel(
                                        current =>
                                            !current
                                    )
                                }

                                style={{

                                    marginTop:
                                        "4px",

                                    padding:
                                        0,

                                    border:
                                        "none",

                                    background:
                                        "transparent",

                                    color:
                                        "rgba(255,255,255,0.65)",

                                    fontSize:
                                        "11px",

                                    cursor:
                                        "pointer"

                                }}
                            >

                                {
                                    showAllFeel
                                    ?
                                    "Show strongest 5"
                                    :
                                    "Show all 13"
                                }

                            </button>


                            <div
                                style={{

                                    marginTop:
                                        "12px",

                                    fontSize:
                                        "10px",

                                    lineHeight:
                                        1.45,

                                    opacity:
                                        0.38

                                }}
                            >

                                {
                                    atlasContext.isFeelAtlas
                                    ?
                                    (
                                        "These standardized experiential "
                                        +
                                        "dimensions form the representation "
                                        +
                                        "used to construct this Feel atlas."
                                    )
                                    :
                                    (
                                        "Experiential profile shown for "
                                        +
                                        "context; these dimensions were not "
                                        +
                                        "used to position this item in the "
                                        +
                                        "current atlas."
                                    )
                                }

                            </div>

                        </>
                    )
                }

            </Section>


            {/* =================================================
                REGION
            ================================================= */}

            <Section
                title="Region"
            >

                {
                    clusterStats
                    ?
                    (
                        <>

                            <div
                                style={{

                                    fontSize:
                                        "14px",

                                    fontWeight:
                                        600,

                                    lineHeight:
                                        1.4

                                }}
                            >
                                {
                                    clusterStats.clusterLabel
                                    ??
                                    `Cluster ${clusterStats.clusterId}`
                                }
                            </div>


                            <div
                                style={{

                                    marginTop:
                                        "3px",

                                    marginBottom:
                                        "14px",

                                    fontSize:
                                        "10px",

                                    opacity:
                                        0.4

                                }}
                            >
                                Cluster {
                                    clusterStats.clusterId
                                } · {
                                    formatNumber(
                                        clusterStats.clusterSize
                                    )
                                } items
                            </div>


                            <StatRow

                                label={
                                    `Popularity in region (${domainLabel(
                                        node.domain
                                    )})`
                                }

                                value={
                                    formatTopPercent(
                                        clusterStats
                                            .popularityTopPercent
                                    )
                                }

                            />


                            <StatRow

                                label="2D centrality"

                                value={
                                    `${formatTopPercent(
                                        clusterStats
                                            .centralityTopPercent
                                    )} most central`
                                }

                            />


                            <StatRow

                                label="Region position"

                                value={
                                    clusterStats
                                        .boundaryStatus
                                }

                            />


                            {
                                clusterStats
                                    .nearestCompetingCluster
                                &&
                                (
                                    <StatRow

                                        label={
                                            "Nearest region"
                                        }

                                        value={
                                            clusterStats
                                                .nearestCompetingCluster
                                                .clusterLabel
                                            ??
                                            `Cluster ${
                                                clusterStats
                                                    .nearestCompetingCluster
                                                    .clusterId
                                            }`
                                        }

                                    />
                                )
                            }


                            {
                                clusterStats
                                    .currentClusterProximity
                                !== null
                                &&
                                clusterStats
                                    .competingClusterProximity
                                !== null
                                &&
                                (
                                    <div
                                        style={{
                                            marginTop: "14px"
                                        }}
                                    >

                                        <div
                                            style={{

                                                marginBottom:
                                                    "7px",

                                                fontSize:
                                                    "10px",

                                                opacity:
                                                    0.42

                                            }}
                                        >
                                            RELATIVE CENTROID PROXIMITY
                                        </div>


                                        <StatRow

                                            label={
                                                "Current region"
                                            }

                                            value={
                                                `${Math.round(
                                                    clusterStats
                                                        .currentClusterProximity
                                                        *
                                                        100
                                                )}%`
                                            }

                                        />


                                        <StatRow

                                            label={
                                                "Nearest region"
                                            }

                                            value={
                                                `${Math.round(
                                                    clusterStats
                                                        .competingClusterProximity
                                                        *
                                                        100
                                                )}%`
                                            }

                                        />


                                        <div
                                            style={{

                                                marginTop:
                                                    "4px",

                                                fontSize:
                                                    "9px",

                                                lineHeight:
                                                    1.4,

                                                opacity:
                                                    0.32

                                            }}
                                        >
                                            Relative distance to cluster
                                            centroids; not a membership
                                            probability.
                                        </div>

                                    </div>
                                )
                            }


                            {
                                clusterStats
                                    .domainComposition
                                    .length > 1
                                &&
                                (
                                    <div
                                        style={{
                                            marginTop: "15px"
                                        }}
                                    >

                                        <div
                                            style={{

                                                marginBottom:
                                                    "8px",

                                                fontSize:
                                                    "10px",

                                                opacity:
                                                    0.42

                                            }}
                                        >
                                            DOMAIN COMPOSITION
                                        </div>


                                        {
                                            clusterStats
                                                .domainComposition
                                                .map(

                                                    item => (

                                                        <StatRow

                                                            key={
                                                                item.domain
                                                            }

                                                            label={
                                                                domainLabel(
                                                                    item.domain
                                                                )
                                                            }

                                                            value={
                                                                `${Math.round(
                                                                    item.share
                                                                    *
                                                                    100
                                                                )}%`
                                                            }

                                                        />

                                                    )

                                                )
                                        }

                                    </div>
                                )
                            }

                        </>
                    )
                    :
                    (
                        <div
                            style={{
                                fontSize: "12px",
                                opacity: 0.5
                            }}
                        >
                            This item is not assigned to a
                            standard atlas region.
                        </div>
                    )
                }

            </Section>


            {/* =================================================
                DISCOVERY
            ================================================= */}

            {
                neighborAnalysis
                &&
                (
                    neighborAnalysis.closestMainstream
                    ||
                    neighborAnalysis.closestNiche
                )
                &&
                (
                    <Section
                        title="Discover"
                    >

                        {
                            neighborAnalysis.closestMainstream
                            &&
                            (
                                <div
                                    style={{
                                        marginBottom: "13px"
                                    }}
                                >

                                    <div
                                        style={{

                                            marginBottom:
                                                "5px",

                                            fontSize:
                                                "10px",

                                            opacity:
                                                0.42

                                        }}
                                    >
                                        CLOSEST MAINSTREAM ANALOGUE
                                    </div>


                                    <ClickableNode

                                        result={
                                            neighborAnalysis
                                                .closestMainstream
                                        }

                                        onSelectNode={
                                            onSelectNode
                                        }

                                    />

                                </div>
                            )
                        }


                        {
                            neighborAnalysis.closestNiche
                            &&
                            (
                                <div>

                                    <div
                                        style={{

                                            marginBottom:
                                                "5px",

                                            fontSize:
                                                "10px",

                                            opacity:
                                                0.42

                                        }}
                                    >
                                        CLOSEST NICHE ANALOGUE
                                    </div>


                                    <ClickableNode

                                        result={
                                            neighborAnalysis
                                                .closestNiche
                                        }

                                        onSelectNode={
                                            onSelectNode
                                        }

                                    />

                                </div>
                            )
                        }


                        <div
                            style={{

                                marginTop:
                                    "8px",

                                fontSize:
                                    "9px",

                                lineHeight:
                                    1.4,

                                opacity:
                                    0.32

                            }}
                        >
                            Mainstream and niche are defined
                            using the top and bottom 20% of
                            popularity within each item's
                            own domain.
                        </div>

                    </Section>
                )
            }


            {/* =================================================
                MORE ITEM DATA
            ================================================= */}

            <Section
                title="More item data"
            >

                <button
                    type="button"

                    onClick={
                        handleToggleMore
                    }

                    style={{

                        width:
                            "100%",

                        padding:
                            "9px 11px",

                        border:
                            "1px solid rgba(255,255,255,0.09)",

                        borderRadius:
                            "6px",

                        background:
                            "rgba(255,255,255,0.03)",

                        color:
                            "white",

                        cursor:
                            "pointer",

                        textAlign:
                            "left",

                        fontSize:
                            "12px"

                    }}
                >

                    {
                        moreExpanded
                        ?
                        "Hide item data ▴"
                        :
                        "Expand item data ▾"
                    }

                </button>


                {
                    moreExpanded
                    &&
                    (
                        <div
                            style={{
                                marginTop: "16px"
                            }}
                        >


                            {/* ---------------------------------
                                CATEGORIES / GENRES
                            --------------------------------- */}

                            {
                                categories.length > 0
                                &&
                                (
                                    <div
                                        style={{
                                            marginBottom: "18px"
                                        }}
                                    >

                                        <div
                                            style={{

                                                marginBottom:
                                                    "8px",

                                                fontSize:
                                                    "10px",

                                                fontWeight:
                                                    700,

                                                opacity:
                                                    0.45

                                            }}
                                        >
                                            CATEGORIES / GENRES
                                        </div>


                                        <div
                                            style={{

                                                display:
                                                    "flex",

                                                flexWrap:
                                                    "wrap",

                                                gap:
                                                    "5px"

                                            }}
                                        >

                                            {
                                                categories.map(

                                                    category => (

                                                        <span
                                                            key={
                                                                category
                                                            }

                                                            style={{

                                                                padding:
                                                                    "4px 7px",

                                                                borderRadius:
                                                                    "999px",

                                                                background:
                                                                    "rgba(255,255,255,0.055)",

                                                                fontSize:
                                                                    "10px",

                                                                opacity:
                                                                    0.75

                                                            }}
                                                        >
                                                            {category}
                                                        </span>

                                                    )

                                                )
                                            }

                                        </div>

                                    </div>
                                )
                            }


                            {/* ---------------------------------
                                ACTORS / OTHER METADATA
                            --------------------------------- */}

                            {
                                node.metadata.actors
                                &&
                                node.metadata.actors.length > 0
                                &&
                                (
                                    <div
                                        style={{
                                            marginBottom: "18px"
                                        }}
                                    >

                                        <div
                                            style={{

                                                marginBottom:
                                                    "6px",

                                                fontSize:
                                                    "10px",

                                                fontWeight:
                                                    700,

                                                opacity:
                                                    0.45

                                            }}
                                        >
                                            ACTORS
                                        </div>


                                        <div
                                            style={{

                                                fontSize:
                                                    "11px",

                                                lineHeight:
                                                    1.55,

                                                opacity:
                                                    0.68

                                            }}
                                        >
                                            {
                                                node.metadata
                                                    .actors
                                                    .join(", ")
                                            }
                                        </div>

                                    </div>
                                )
                            }


                            {/* ---------------------------------
                                TAGS / SEMANTIC TEXT
                            --------------------------------- */}

                            {
                                node.text.tags.length > 0
                                &&
                                (
                                    <div
                                        style={{
                                            marginBottom: "18px"
                                        }}
                                    >

                                        <div
                                            style={{

                                                marginBottom:
                                                    "6px",

                                                fontSize:
                                                    "10px",

                                                fontWeight:
                                                    700,

                                                opacity:
                                                    0.45

                                            }}
                                        >
                                            TAGS / SEMANTIC TEXT
                                        </div>


                                        <div
                                            style={{

                                                maxHeight:
                                                    "180px",

                                                overflowY:
                                                    "auto",

                                                padding:
                                                    "10px",

                                                borderRadius:
                                                    "6px",

                                                background:
                                                    "rgba(255,255,255,0.03)",

                                                whiteSpace:
                                                    "pre-wrap",

                                                overflowWrap:
                                                    "anywhere",

                                                fontSize:
                                                    "10px",

                                                lineHeight:
                                                    1.5,

                                                opacity:
                                                    0.62

                                            }}
                                        >
                                            {
                                                node.text.tags
                                                    .join("\n\n")
                                            }
                                        </div>

                                    </div>
                                )
                            }


                            {/* ---------------------------------
                                SEMANTIC COVERAGE
                            --------------------------------- */}

                            {
                                profile
                                &&
                                (
                                    <div
                                        style={{
                                            marginBottom: "18px"
                                        }}
                                    >

                                        <div
                                            style={{

                                                marginBottom:
                                                    "8px",

                                                fontSize:
                                                    "10px",

                                                fontWeight:
                                                    700,

                                                opacity:
                                                    0.45

                                            }}
                                        >
                                            REPRESENTATION DATA
                                        </div>


                                        <StatRow

                                            label={
                                                "Base semantics"
                                            }

                                            value={
                                                profile.semantic
                                                    .has_base_semantics
                                                ?
                                                "Yes"
                                                :
                                                "No"
                                            }

                                        />


                                        <StatRow

                                            label={
                                                "Review semantics"
                                            }

                                            value={
                                                profile.semantic
                                                    .has_review_semantics
                                                ?
                                                "Yes"
                                                :
                                                "No"
                                            }

                                        />


                                        <StatRow

                                            label={
                                                "Reviews available"
                                            }

                                            value={
                                                formatNumber(
                                                    profile.reviews
                                                        .available_count
                                                )
                                            }

                                        />


                                        <StatRow

                                            label={
                                                "Reviews used for embedding"
                                            }

                                            value={
                                                formatNumber(
                                                    profile.reviews
                                                        .used_for_embedding
                                                )
                                            }

                                        />

                                    </div>
                                )
                            }


                            {/* ---------------------------------
                                REVIEWS
                            --------------------------------- */}

                            <div>

                                <div
                                    style={{

                                        marginBottom:
                                            "8px",

                                        fontSize:
                                            "10px",

                                        fontWeight:
                                            700,

                                        opacity:
                                            0.45

                                    }}
                                >
                                    REVIEWS
                                </div>


                                {
                                    reviewsLoading
                                    &&
                                    (
                                        <div
                                            style={{
                                                fontSize: "11px",
                                                opacity: 0.45
                                            }}
                                        >
                                            Loading review samples...
                                        </div>
                                    )
                                }


                                {
                                    reviewsError
                                    &&
                                    (
                                        <div
                                            style={{
                                                fontSize: "11px",
                                                opacity: 0.55
                                            }}
                                        >
                                            Review samples could not be loaded.
                                        </div>
                                    )
                                }


                                {
                                    reviewsLoaded
                                    &&
                                    reviews
                                    &&
                                    reviews.reviews.length === 0
                                    &&
                                    (
                                        <div
                                            style={{
                                                fontSize: "11px",
                                                opacity: 0.5
                                            }}
                                        >
                                            No review text is available
                                            for this item.
                                        </div>
                                    )
                                }


                                {
                                    reviews
                                    &&
                                    reviews.reviews.length > 0
                                    &&
                                    (
                                        <>

                                            <div
                                                style={{

                                                    marginBottom:
                                                        "9px",

                                                    fontSize:
                                                        "10px",

                                                    opacity:
                                                        0.4

                                                }}
                                            >
                                                {
                                                    formatNumber(
                                                        reviews
                                                            .available_count
                                                    )
                                                } available · {
                                                    reviews
                                                        .sampled_count
                                                } shown
                                            </div>


                                            <ReviewViewer

                                                review={
                                                    reviews
                                                        .reviews[
                                                            reviewIndex
                                                        ]
                                                }

                                                index={
                                                    reviewIndex
                                                }

                                                count={
                                                    reviews
                                                        .reviews
                                                        .length
                                                }

                                                onPrevious={() =>

                                                    setReviewIndex(

                                                        current =>

                                                            (
                                                                current
                                                                -
                                                                1
                                                                +
                                                                reviews
                                                                    .reviews
                                                                    .length
                                                            )
                                                            %
                                                            reviews
                                                                .reviews
                                                                .length

                                                    )

                                                }

                                                onNext={() =>

                                                    setReviewIndex(

                                                        current =>

                                                            (
                                                                current
                                                                +
                                                                1
                                                            )
                                                            %
                                                            reviews
                                                                .reviews
                                                                .length

                                                    )

                                                }

                                            />


                                            <div
                                                style={{

                                                    marginTop:
                                                        "10px",

                                                    fontSize:
                                                        "9px",

                                                    lineHeight:
                                                        1.45,

                                                    opacity:
                                                        0.3

                                                }}
                                            >
                                                Reviews shown here are
                                                deterministic inspection
                                                samples and are not claimed
                                                to be the exact subset used
                                                when the pooled review
                                                embedding was created.
                                            </div>

                                        </>
                                    )
                                }

                            </div>

                        </div>
                    )
                }

            </Section>

        </aside>

    );

}