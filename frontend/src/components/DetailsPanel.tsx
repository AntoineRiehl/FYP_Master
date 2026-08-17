// frontend/src/components/DetailsPanel.tsx

import type {
    AtlasNode
} from "../types/atlas";


type Props = {

    node: AtlasNode | null;

};



export default function DetailsPanel({

    node

}: Props) {


    return (

        <aside

            style={{

                width:"320px",

                padding:"20px",

                background:"#0b1020",

                borderLeft:
                    "1px solid rgba(255,255,255,0.1)",

                boxSizing:"border-box",

                overflowY:"auto"

            }}

        >


            <h2

                style={{

                    marginTop:0,

                    fontSize:"20px"

                }}

            >

                Details

            </h2>



            {!node ? (

                <div

                    style={{

                        marginTop:"30px",

                        opacity:0.5,

                        fontSize:"14px"

                    }}

                >

                    Select a node to inspect it.

                </div>


            ) : (


                <>


                    <h3>

                        {node.title}

                    </h3>



                    <p>

                        <strong>
                            Domain:
                        </strong>{" "}

                        {node.domain}

                    </p>



                    <p>

                        <strong>
                            Cluster:
                        </strong>{" "}

                        {
                            node.visual.cluster_label
                            ??
                            node.visual.cluster
                            ??
                            "Unknown"
                        }

                    </p>



                    {
                        node.metadata.year && (

                            <p>

                                <strong>
                                    Year:
                                </strong>{" "}

                                {node.metadata.year}

                            </p>

                        )
                    }



                    {
                        node.metadata.director && (

                            <p>

                                <strong>
                                    Director:
                                </strong>{" "}

                                {node.metadata.director}

                            </p>

                        )
                    }



                    {
                        node.statistics.rating !== null
                        &&
                        node.statistics.rating !== undefined
                        && (

                            <p>

                                <strong>
                                    Rating:
                                </strong>{" "}

                                {
                                    node.statistics.rating
                                        .toFixed(2)
                                }

                            </p>

                        )
                    }



                    {
                        node.statistics.rating_count !== null
                        &&
                        node.statistics.rating_count !== undefined
                        && (

                            <p>

                                <strong>
                                    Ratings:
                                </strong>{" "}

                                {
                                    node.statistics.rating_count
                                }

                            </p>

                        )
                    }



                    {
                        node.statistics.popularity !== null
                        &&
                        node.statistics.popularity !== undefined
                        && (

                            <p>

                                <strong>
                                    Popularity:
                                </strong>{" "}

                                {
                                    Math.round(
                                        node.statistics.popularity
                                    )
                                }

                            </p>

                        )
                    }



                    {
                        node.text.categories.length > 0 && (

                            <div>

                                <strong>
                                    Categories:
                                </strong>


                                <p>

                                    {
                                        node.text.categories.join(
                                            ", "
                                        )
                                    }

                                </p>

                            </div>

                        )
                    }



                    {
                        node.text.tags.length > 0 && (

                            <div>

                                <strong>
                                    Tags:
                                </strong>


                                <p>

                                    {
                                        node.text.tags.join(
                                            ", "
                                        )
                                    }

                                </p>

                            </div>

                        )
                    }



                    {
                        node.metadata.actors &&
                        node.metadata.actors.length > 0
                        && (

                            <div>

                                <strong>
                                    Actors:
                                </strong>


                                <p>

                                    {
                                        node.metadata.actors.join(
                                            ", "
                                        )
                                    }

                                </p>

                            </div>

                        )
                    }



                    {
                        Object.keys(
                            node.enrichment ?? {}
                        ).length > 0 && (

                            <div>

                                <strong>
                                    Enrichment:
                                </strong>


                                <pre

                                    style={{

                                        whiteSpace:
                                            "pre-wrap",

                                        fontSize:
                                            "12px",

                                        opacity:
                                            0.8

                                    }}

                                >

                                    {
                                        JSON.stringify(
                                            node.enrichment,
                                            null,
                                            2
                                        )
                                    }

                                </pre>

                            </div>

                        )
                    }


                </>


            )}


        </aside>

    );

}