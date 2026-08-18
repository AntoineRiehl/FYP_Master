// frontend/src/components/sidebar/SearchPanel.tsx

import type {
    AtlasNode
} from "../../types/atlas";


// =====================================================
// PROPS
// =====================================================

type Props = {

    searchQuery: string;

    onSearchChange:
        (query: string) => void;

    searchResults: AtlasNode[];

    onSelectResult:
        (node: AtlasNode) => void;

};


// =====================================================
// COMPONENT
// =====================================================

export default function SearchPanel({

    searchQuery,

    onSearchChange,

    searchResults,

    onSelectResult

}: Props) {

    return (

        <section
            style={{
                marginTop: "30px"
            }}
        >

            <h3
                style={{
                    fontSize: "14px",
                    opacity: 0.7
                }}
            >
                Search
            </h3>


            <input

                value={searchQuery}

                onChange={(e) =>
                    onSearchChange(
                        e.target.value
                    )
                }

                placeholder="Search..."

                style={{
                    width: "100%",
                    padding: "8px",
                    background: "#111827",
                    color: "white",
                    boxSizing: "border-box",
                    border:
                        "1px solid rgba(255,255,255,0.1)",
                    borderRadius: "4px"
                }}

            />


            {
                searchResults.length > 0
                &&
                (

                    <div
                        style={{
                            marginTop: "10px"
                        }}
                    >

                        {
                            searchResults.map(
                                node => (

                                    <div

                                        key={node.id}

                                        onClick={() =>
                                            onSelectResult(
                                                node
                                            )
                                        }

                                        style={{
                                            padding: "6px",
                                            cursor:
                                                "pointer",
                                            borderBottom:
                                                "1px solid rgba(255,255,255,0.1)"
                                        }}

                                    >

                                        {node.title}

                                    </div>

                                )
                            )
                        }

                    </div>

                )
            }

        </section>

    );

}