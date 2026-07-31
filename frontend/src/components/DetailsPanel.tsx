// frontend/src/components/DetailsPanel.tsx

import type { AtlasNode } from "../types/atlas";

type Props = {
    node: AtlasNode | null;
};

export default function DetailsPanel({
    node
}: Props) {

    return (

        <aside
            style={{
                width: "300px",
                padding: "20px",
                background: "#0b1020",
                borderLeft: "1px solid rgba(255,255,255,0.1)",
                boxSizing: "border-box",
                overflowY: "auto"
            }}
        >

            <h2
                style={{
                    marginTop: 0,
                    fontSize: "20px"
                }}
            >
                Details
            </h2>

            {!node ? (

                <div
                    style={{
                        marginTop: "30px",
                        opacity: 0.5,
                        fontSize: "14px"
                    }}
                >
                    Hover a node to inspect it.
                </div>

            ) : (

                <>

                    <h3>{node.label}</h3>

                    <p>
                        <strong>Cluster:</strong>{" "}
                        {node.cluster_label ?? node.cluster}
                    </p>

                    {node.category && (

                        <p>

                            <strong>Category:</strong>{" "}
                            {node.category}

                        </p>

                    )}

                    {node.popularity !== undefined && (

                        <p>

                            <strong>Popularity:</strong>{" "}
                            {Math.round(node.popularity)}

                        </p>

                    )}

                    {node.rating !== undefined && (

                        <p>

                            <strong>Rating:</strong>{" "}
                            {node.rating.toFixed(2)}

                        </p>

                    )}

                    {node.description && (

                        <p
                            style={{
                                marginTop: "20px",
                                opacity: 0.8
                            }}
                        >
                            {node.description}
                        </p>

                    )}

                </>

            )}

        </aside>

    );

}