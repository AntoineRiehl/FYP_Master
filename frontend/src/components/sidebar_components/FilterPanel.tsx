// frontend/src/components/sidebar/FilterPanel.tsx

import type {
    AtlasFilters
} from "../../types/filters";


// =====================================================
// PROPS
// =====================================================

type Props = {

    filters: AtlasFilters;

    availableDomains: string[];

    onFiltersChange:
        (filters: AtlasFilters) => void;

};


// =====================================================
// COMPONENT
// =====================================================

export default function FilterPanel({

    filters,

    availableDomains,

    onFiltersChange

}: Props) {


    // =================================================
    // DOMAIN TOGGLE
    // =================================================

    function toggleDomain(
        domain: string
    ) {

        const selected =
            filters.domains.includes(
                domain
            );


        const domains =

            selected

            ?

            filters.domains.filter(
                current =>
                    current !== domain
            )

            :

            [
                ...filters.domains,
                domain
            ];


        onFiltersChange({

            ...filters,

            domains

        });

    }


    // =================================================
    // RESET
    // =================================================

    function resetFilters() {

        onFiltersChange({

            domains:
                availableDomains,

            reviewsOnly:
                false,

            balanceMode:
                "all"

        });

    }


    // =================================================
    // RENDER
    // =================================================

    return (

        <section
            style={{
                marginTop: "30px"
            }}
        >

            {/* HEADER */}

            <div
                style={{
                    display: "flex",
                    justifyContent:
                        "space-between",
                    alignItems: "center"
                }}
            >

                <h3
                    style={{
                        fontSize: "14px",
                        opacity: 0.7,
                        margin: 0
                    }}
                >
                    Filters
                </h3>


                <button
                    onClick={resetFilters}

                    style={{
                        background:
                            "transparent",
                        border: "none",
                        color: "#9ca3af",
                        cursor: "pointer",
                        fontSize: "12px"
                    }}
                >
                    Reset
                </button>

            </div>


            {/* =================================================
                DOMAINS
            ================================================= */}

            <div
                style={{
                    marginTop: "15px"
                }}
            >

                <div
                    style={{
                        fontSize: "13px",
                        marginBottom: "8px",
                        opacity: 0.8
                    }}
                >
                    Domains
                </div>


                {
                    availableDomains.map(
                        domain => (

                            <label
                                key={domain}

                                style={{
                                    display: "flex",
                                    alignItems:
                                        "center",
                                    gap: "8px",
                                    marginBottom:
                                        "7px",
                                    fontSize:
                                        "13px",
                                    cursor:
                                        "pointer"
                                }}
                            >

                                <input
                                    type="checkbox"

                                    checked={
                                        filters.domains
                                            .includes(
                                                domain
                                            )
                                    }

                                    onChange={() =>
                                        toggleDomain(
                                            domain
                                        )
                                    }
                                />

                                <span>
                                    {domain}
                                </span>

                            </label>

                        )
                    )
                }

            </div>


            {/* =================================================
                REVIEWS
            ================================================= */}

            <div
                style={{
                    marginTop: "20px"
                }}
            >

                <label
                    style={{
                        display: "flex",
                        alignItems:
                            "center",
                        gap: "8px",
                        fontSize: "13px",
                        cursor:
                            "pointer"
                    }}
                >

                    <input

                        type="checkbox"

                        checked={
                            filters.reviewsOnly
                        }

                        onChange={(e) =>

                            onFiltersChange({

                                ...filters,

                                reviewsOnly:
                                    e.target.checked

                            })

                        }

                    />

                    <span>
                        Only items with reviews
                    </span>

                </label>

            </div>


            {/* =================================================
                DOMAIN BALANCE
            ================================================= */}

            <div
                style={{
                    marginTop: "20px"
                }}
            >

                <div
                    style={{
                        fontSize: "13px",
                        marginBottom: "8px",
                        opacity: 0.8
                    }}
                >
                    Domain balance
                </div>


                <select

                    value={
                        filters.balanceMode
                    }

                    onChange={(e) => {

                        const value =
                            e.target.value;

                        if (
                            value === "all"
                            ||
                            value === "balanced"
                        ) {

                            onFiltersChange({

                                ...filters,

                                balanceMode:
                                    value

                            });

                        }

                    }}

                    style={{
                        width: "100%",
                        padding: "7px",
                        background: "#111827",
                        color: "white",
                        border:
                            "1px solid rgba(255,255,255,0.1)",
                        borderRadius: "4px"
                    }}

                >

                    <option value="all">
                        All items
                    </option>

                    <option value="balanced">
                        Balanced domains
                    </option>

                </select>

            </div>

        </section>

    );

}