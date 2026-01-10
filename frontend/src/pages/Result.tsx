import { useEffect, useState } from "react";
import type { PredictionResult } from "../types/prediction";
import { Box } from "@mui/material";

import DiagnosisCard from "../components/DiagnosisCard";
import ConfidenceBar from "../components/ConfidenceBar";
import ProbabilityChart from "../components/ProbabilityChart";
import GradCamOverlay from "../components/GradCamOverlay";
import Loader from "../components/Loader";
import DiagnosisReview from "../components/DiagnosisReview";
import ReviewCard from "../components/ReviewCard";

export default function Result() {
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [image, setImage] = useState<string | null>(null);
  const [historyItem, setHistoryItem] = useState<any>(null);
  const [editingReview, setEditingReview] = useState(false);

  useEffect(() => {
    const storedPrediction = localStorage.getItem("prediction");
    const storedImage = localStorage.getItem("uploaded_image");
    const rawHistory = localStorage.getItem("history");

    if (!storedPrediction || !storedImage || !rawHistory) return;

    const parsedResult = JSON.parse(storedPrediction);
    const parsedHistory = JSON.parse(rawHistory);

    const current = parsedHistory.find(
      (item: any) =>
        item.image === storedImage &&
        item.result.class === parsedResult.class
    );

    setResult(parsedResult);
    setImage(storedImage);
    setHistoryItem(current);
  }, []);

  if (!result || !image || !historyItem) {
    return <Loader />;
  }

  const isPinned = historyItem.reviewPinned === true;

  const handleSaveReview = (review: any) => {
    const raw = localStorage.getItem("history");
    const history = raw ? JSON.parse(raw) : [];

    const updated = history.map((item: any) =>
      item.id === historyItem.id
        ? {
            ...item,
            review,
            reviewPinned: true, // 🔒 se fija para siempre
          }
        : item
    );

    localStorage.setItem("history", JSON.stringify(updated));

    setHistoryItem({
      ...historyItem,
      review,
      reviewPinned: true,
    });

    setEditingReview(false);
  };

  const ReviewComponent =
    !historyItem.review || editingReview ? (
      <DiagnosisReview
        detectedClass={result.class}
        initialReview={historyItem.review}
        onSave={handleSaveReview}
      />
    ) : (
      <ReviewCard
        review={historyItem.review}
        onEdit={() => setEditingReview(true)}
      />
    );

  return (
    <>
      <DiagnosisCard result={result} />

      {/* POSICIÓN 1 */}
      {isPinned && ReviewComponent}

      <ConfidenceBar confidence={result.confidence} />
      <ProbabilityChart probabilities={result.probabilities} />

      <Box align="center" >
        <GradCamOverlay
        image={image}
        heatmap={
          result.heatmap_url
            ? result.heatmap_url.startsWith("data:image")
              ? result.heatmap_url
              : `data:image/png;base64,${result.heatmap_url}`
            : null
        }
      />
      </Box>
        
      
      {/* POSICIÓN 2 */}
      {!isPinned && ReviewComponent}
    </>
  );
}
