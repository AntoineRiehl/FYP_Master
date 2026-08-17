// frontend/src/components/Sidebar.tsx

import type { AtlasNode } from "../types/atlas";


type Props = {

    atlasName:string;

    onAtlasChange:
        (name:string)=>void;


    searchQuery:string;

    onSearchChange:
        (query:string)=>void;


    searchResults:AtlasNode[];

    onSelectResult:
        (node:AtlasNode)=>void;

};



export default function Sidebar({

    atlasName,
    onAtlasChange,

    searchQuery,
    onSearchChange,

    searchResults,
    onSelectResult

}:Props) {


    return (

        <aside
            style={{
                width:"260px",
                padding:"20px",
                background:"#0b1020",
                borderRight:
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
                Atlas Explorer
            </h2>



            <section
                style={{
                    marginTop:"30px"
                }}
            >

                <h3
                    style={{
                        fontSize:"14px",
                        opacity:0.7
                    }}
                >
                    Atlas
                </h3>



                <select

                    value={atlasName}

                    onChange={(e)=>
                        onAtlasChange(
                            e.target.value
                        )
                    }

                    style={{
                        width:"100%",
                        padding:"8px",
                        background:"#111827",
                        color:"white"
                    }}

                >

                    <option value="movies">
                        Movies
                    </option>

                    <option value="music">
                        Music
                    </option>

                    <option value="restaurants">
                        Restaurants
                    </option>


                </select>


            </section>



            <section
                style={{
                    marginTop:"30px"
                }}
            >

                <h3
                    style={{
                        fontSize:"14px",
                        opacity:0.7
                    }}
                >
                    Search
                </h3>


                <input

                    value={searchQuery}

                    onChange={(e)=>
                        onSearchChange(
                            e.target.value
                        )
                    }

                    placeholder="Search..."

                    style={{
                        width:"100%",
                        padding:"8px",
                        background:"#111827",
                        color:"white",
                        boxSizing:"border-box"
                    }}

                />



                {
                    searchResults.length > 0 && (

                        <div
                            style={{
                                marginTop:"10px"
                            }}
                        >

                            {
                                searchResults.map(node=>(

                                    <div

                                        key={node.id}

                                        onClick={()=>
                                            onSelectResult(node)
                                        }

                                        style={{
                                            padding:"6px",
                                            cursor:"pointer",
                                            borderBottom:
                                                "1px solid rgba(255,255,255,0.1)"
                                        }}

                                    >

                                        {node.title}

                                    </div>

                                ))

                            }

                        </div>

                    )
                }


            </section>



        </aside>

    );

}