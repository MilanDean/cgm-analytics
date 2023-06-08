import Image from 'next/image';

export default function TeamCard(props: {name: string, title: string}) {
    return (

        <main>
            <div className="flex justify-center h-full w-full">

                <div className='flex flex-col h-72 w-72'>
                    <div className='shadow-2xl rounded-lg border-2 border-gray-200'>
                        <Image alt="image" src="/stockphoto.jpg" width={300} height={500} priority/>
                        <div className='flex flex-col bg-black text-white justify-center items-center mt-auto rounded-b-md border-2 border-black'>
                            <p> {props.name} </p>
                            <p> {props.title} </p>
                        </div>
                    </div>

                </div>
        
            </div>
        </main>

    )
}