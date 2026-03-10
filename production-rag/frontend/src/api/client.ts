import axios from 'axios';
import type { ApiResponse, HealthData, IngestResult, QueryResult } from '../types';

const baseURL = import.meta.env.VITE_API_URL ?? 'http://localhost:8000';

const api = axios.create({ baseURL, timeout: 60000 });

export async function fetchHealth(): Promise<HealthData> {
  const { data } = await api.get<ApiResponse<HealthData>>('/health');
  return data.data;
}

export async function askQuestion(query: string, topK: number): Promise<QueryResult> {
  const { data } = await api.post<ApiResponse<QueryResult>>('/query', {
    query,
    top_k: topK,
  });
  return data.data;
}

export async function seedSamples(): Promise<IngestResult> {
  const { data } = await api.post<ApiResponse<IngestResult>>('/ingest/seed');
  return data.data;
}

export async function ingestFiles(files: File[]): Promise<IngestResult> {
  const form = new FormData();
  files.forEach((file) => form.append('files', file));
  const { data } = await api.post<ApiResponse<IngestResult>>('/ingest', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  return data.data;
}
