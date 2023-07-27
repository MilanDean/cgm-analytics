import React from 'react';
import Link from 'next/link';

const TopNav: React.FC = () => (
  <header className="text-gray-600 body-font w-full">
    <div className="container mx-auto flex flex-wrap p-5 flex-col md:flex-row items-center">
      <Link href="/">
        <div className="mb-4 md:mb-0">
            <img
              className="w-20"
              src='./HomeLogo.png'
              alt="Nutrinet"
            />
        </div>
      </Link>
      <nav className="md:ml-auto flex px-4 sm:px-6 lg:px-8">
        <Link href="/upload">
          <span className="mr-5 hover:text-gray-900">Upload</span>
        </Link>
        <Link href="/analysis">
          <span className="mr-5 hover:text-gray-900">Analysis</span>
        </Link>
        <Link href="/team">
          <span className="mr-5 hover:text-gray-900">Team</span>
        </Link>
      </nav>
    </div>
  </header>
);

export default TopNav;
