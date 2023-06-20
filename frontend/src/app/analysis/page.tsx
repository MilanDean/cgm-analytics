'use client';

import { useEffect, useState } from 'react';
import { v4 as uuidv4 } from 'uuid';
import axios from 'axios';
import Image from 'next/image';
import TopNav from '../components/TopNav';
import classNames from 'classnames';

export default function Analysis() {
  const [data, setData] = useState([]);
  const [graphIds, setGraphIds] = useState([]);

  useEffect(() => {
      fetchData();
  }, []);

  const fetchData = async () => {
      try {
          const response = await axios.get('http://127.0.0.1:8000/api/analysis');
          setData(response.data);

          const graphResponse = await axios.post('http://127.0.0.1:8000/api/generate_graph');
          setGraphIds(graphResponse.data.graph_ids);
      } catch (error) {
          console.error(error);
      }
  };

  const renderGraphs = () => {
    return graphIds.map((graphId) => (
        <Image
            key={graphId}
            src={`http://127.0.0.1:8000/api/graph/${graphId}`}
            alt={`Graph ${graphId}`}
            className="dark:invert mx-3"
            width={1200}
            height={400}
            priority
        />
    ));
  };

const columns = Object.keys(data[0] || {});
  
  return (
    <div className="px-4 sm:px-6 lg:px-8">
      <TopNav />
      <div className="mt-8 flow-root">
        <div className="-mx-4 -my-2 sm:-mx-6 lg:-mx-8 border-2 border-black justify-center items-center">
          <header className='p-5'> 
            <div className='text-center text-2xl font-mono font-bold'>
              <h1> Analytics Dashboard </h1>
            </div>
          </header>
          <div className="flex md:table-fixed overflow-x-auto h-96 overflow-y-auto py-2">
            <table className="p-5 border-separate border-spacing-0">
              <thead>
                <tr>
                  {columns.map((column) => (
                    <th
                      key={column}
                      scope="col"
                      className="sticky top-0 z-10 border-b border-gray-300 bg-white bg-opacity-75 py-3.5 pl-4 pr-3 text-left text-sm font-semibold text-gray-900 backdrop-blur backdrop-filter sm:pl-6 lg:pl-8"
                    >
                      {column}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {data.map((record) => (
                  <tr key={uuidv4()}>
                    {columns.map((column) => (
                      <td
                        key={column}
                        className={classNames(
                          'border-b border-gray-200',
                          'whitespace-nowrap py-4 pl-4 pr-3 text-sm font-medium text-gray-900 sm:pl-6 lg:pl-8'
                        )}
                      >
                        {record[column]}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="flex p-3 h-1/2 justify-center items-center">
            {renderGraphs()}
          </div>
        </div>
      </div>
    </div>
  )
};
