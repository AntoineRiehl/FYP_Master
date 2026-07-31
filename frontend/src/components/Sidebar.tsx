// frontend/src/components/Sidebar.tsx

type Props = {

    atlasName:string;

    onAtlasChange:
        (name:string)=>void;

};



export default function Sidebar({
    atlasName,
    onAtlasChange
}:Props) {


    return (

        <aside
            style={{
                width:"260px",
                padding:"20px",
                background:"#0b1020",
                borderRight:
                    "1px solid rgba(255,255,255,0.1)",
                boxSizing:"border-box"
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
                        color:"white",
                        border:
                            "1px solid rgba(255,255,255,0.2)"
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

                    placeholder="Search..."

                    style={{
                        width:"100%",
                        padding:"8px",
                        background:"#111827",
                        color:"white",
                        border:
                            "1px solid rgba(255,255,255,0.2)",
                        boxSizing:"border-box"
                    }}

                />


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
                    Filters
                </h3>


                <p
                    style={{
                        opacity:0.5,
                        fontSize:"13px"
                    }}
                >
                    Filters coming soon...
                </p>


            </section>


        </aside>

    );

}