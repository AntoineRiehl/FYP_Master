//frontend/App.tsx

import { useEffect, useState } from "react";
import UniverseCanvas from "./components/UniverseCanvas";
import type { MovieNode } from "./types/movie";

function App() {
  const [data, setData] = useState<MovieNode[]>([]);

  useEffect(() => {
    fetch("/public/data/movie_map.json")
      .then((res) => res.json())
      .then(setData);
  }, []);

  return (
    <div>
      <UniverseCanvas data={data} />
    </div>
  );
}

export default App;