'use client';

import { useEffect, useState } from 'react';
import { v4 as uuidv4 } from 'uuid';
import axios from 'axios';
import TopNav from '../components/TopNav';
import classNames from 'classnames';
  
  export default function Analysis() {
    const [data, setData] = useState([]);

    useEffect(() => {
        fetchData();
    }, []);

    const fetchData = async () => {
        try {
            const response = await axios.get('http://127.0.0.1:8000/api/analysis');
            setData(response.data);
        } catch (error) {
            console.error(error);
        }
    };

  const columns = Object.keys(data[0] || {});
    
    return (
      <div className="px-4 sm:px-6 lg:px-8">
        <TopNav />
        <div className="mt-8 flow-root">
          <div className="-mx-4 -my-2 sm:-mx-6 lg:-mx-8">
            <div className="inline-block h-96 overflow-y-auto min-w-full py-2 align-middle">
              <table className="md:table-fixed overflow-scroll min-w-full border-separate border-spacing-0">
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
          </div>
        </div>
      </div>
    )
  };
