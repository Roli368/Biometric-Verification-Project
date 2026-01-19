const ResultCard = ({ data }) => {
  if (!data) return null;

  return (
    <div className="p-6 border rounded-xl">
      <p>
        Match:{" "}
        <span className={data.match ? "text-green-400" : "text-red-400"}>
          {data.match ? "YES" : "NO"}
        </span>
      </p>
      <p>Similarity: {(data.similarity * 100).toFixed(2)}%</p>
    </div>
  );
};

export default ResultCard;
