'use client';

import { useEffect, useState, useRef, useCallback } from 'react';
import { v4 as uuidv4 } from 'uuid';
import axios from 'axios';
import TopNav from '../components/TopNav';

function useInterval(callback: () => void, delay: number | null): void {
  const savedCallback = useRef<() => void>();

  useEffect(() => {
    savedCallback.current = callback;
  }, [callback]);

  useEffect(() => {
    let isMounted = true;

    function tick(): void {
      savedCallback.current!();
    }

    if (delay !== null) {
      const intervalId = setInterval(tick, delay);
      return () => {
        clearInterval(intervalId);
        isMounted = false;
      };
    }
  }, [delay]);
}

type RowData = { [key: string]: any };

export default function Analysis(): JSX.Element {
  const [data, setData] = useState<RowData[]>([]);
  const [isDataAvailable, setIsDataAvailable] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [elapsedTime, setElapsedTime] = useState(0);
  const [barPlotUrl, setBarPlotUrl] = useState<string | null>(null);
  const [linePlotUrl, setLinePlotUrl] = useState<string | null>(null);

  // Retrieving the filename stored at /upload local storage
  // Get the filename from local storage
  let filename: string;
  if (typeof window !== 'undefined') {
      filename = window.localStorage.getItem('uploadedFilename') || "";
  }

  const fetchData = useCallback(() => {
    setIsLoading(true);
  
    axios
      .get<RowData[]>(`https://api.nutrinet-ai.com/api/analysis/${encodeURIComponent(filename)}`)
      .then((response) => {
        setData(response.data);
        checkDataAvailability(response.data);
  
        axios
          .get<{ carb_estimate_url: string, prediction_url: string }>(`https://api.nutrinet-ai.com/api/visualization/${encodeURIComponent(filename)}`)
          .then((response) => {
              setBarPlotUrl(response.data.carb_estimate_url);
              setLinePlotUrl(response.data.prediction_url);
              console.log(response.data)
          })
          .catch((error) => {
              console.log(error);
          });
      })
      .catch((error) => {
        console.log(error);
      });
  }, []);
  
  useEffect(() => {
    fetchData();
  }, [fetchData]);
  

  const checkDataAvailability = (responseData: RowData[]): void => {
    if (responseData.length > 0) {
      setIsDataAvailable(true);
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  useInterval(() => {
    if (isLoading) {
      fetchData();
      setElapsedTime((prevElapsedTime) => prevElapsedTime + 3000);
    }
  }, 2000);

  useEffect(() => {
    if (elapsedTime >= 10000) {
      setElapsedTime(0);
      setIsLoading(false);
    }
  }, [elapsedTime]);

  const columns = Object.keys(data[0] || {});

  return (
    <div className="px-4 sm:px-6 lg:px-8">
      <TopNav />
      <div className="mt-8 flow-root">
        <div className="-mx-4 -my-2 sm:-mx-6 lg:-mx-8 justify-center items-center">
          <header className="p-5">
            <div className="text-center text-2xl font-mono font-bold">
              <h1>Analytics Dashboard</h1>
            </div>
          </header>
          {linePlotUrl && (
            <div className="w-full">
              <iframe
                key={'line-plot'}
                src={linePlotUrl}
                title={`Prediction`}
                className="flex"
                width="100%"
                height="500"
              />
            </div>
          )}
          <div className="flex md:table-fixed overflow-x-auto h-96 overflow-y-auto py-2">
            {barPlotUrl && (
              <div className="w-1/2">
                <iframe
                  key={'bar-plot'}
                  src={barPlotUrl}
                  title={`Carb Estimation`}
                  className="flex"
                  width="100%"
                  height="600"
                />
              </div>
            )}
            <div className="w-1/2">
              <table className="p-5 border-separate border-spacing-0 w-full">
                <thead>
                  <tr>
                    {columns.map((column) => (
                      <th
                        key={column}
                        scope="col"
                        className="sticky top-0 z-10 border-b border-gray-300 bg-white bg-opacity-75 py-3.5 pl-4 pr-3 text-left text-sm font-semibold text-gray-900 backdrop-blur-lg"
                      >
                        {column}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {isLoading ? (
                    <tr className="text-center">
                      <td
                        colSpan={columns.length}
                        className="border-b border-gray-200 bg-white px-4 py-3 text-sm text-center font-semibold text-gray-900"
                      >
                        Data loading...
                      </td>
                    </tr>
                  ) : (
                    data.map((row) => (
                      <tr key={uuidv4()}>
                        {columns.map((column) => (
                          <td
                            key={column}
                            className="border-b border-gray-200 bg-white px-4 py-3 text-sm"
                          >
                            {row[column]}
                          </td>
                        ))}
                      </tr>
                    ))
                  )}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>
    </div>
  );

}
