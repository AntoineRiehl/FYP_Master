// frontend/src/types/filters.ts


// =====================================================
// COLOR MODE
// =====================================================

export type ColorMode =
    | "domain"
    | "cluster"
    | "category"
    | "feel";


// =====================================================
// DOMAIN BALANCE MODE
// =====================================================

export type BalanceMode =
    | "all"
    | "balanced";


// =====================================================
// ATLAS FILTERS
// =====================================================

export type AtlasFilters = {

    /**
     * Domains currently visible.
     *
     * Example:
     *
     * ["movies", "music"]
     *
     * An empty array means:
     * "show all available domains".
     */
    domains: string[];


    /**
     * If true, only items containing
     * at least one review are displayed.
     */
    reviewsOnly: boolean;


    /**
     * Controls whether the atlas should
     * preserve the original domain imbalance
     * or display a balanced number of items
     * from each selected domain.
     */
    balanceMode: BalanceMode;

};